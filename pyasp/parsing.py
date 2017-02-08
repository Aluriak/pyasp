"""
Wrapper around the pyasp.ply submodule.

"""
import re
import inspect

import arpeggio as ap

import pyasp.ply.lex as lex
import pyasp.ply.yacc as yacc
from pyasp.constant import OPTIMIZE
from pyasp.term import Term, TermSet


class CollapsableAtomVisitor(ap.PTNodeVisitor):
    """Implement both the grammar and the way to handle it, dedicated to the
    parsing of ASP like string to produce TermSet instances.

    This is a more complex version than AtomVisitor, because implementing
    special use cases, notably partial cut of parsed atoms.

    collapseTerms: function terms in predicate arguments are collapsed into strings
    collapseAtoms: atoms (predicate plus terms) are collapsed into strings
                   requires that collapseTerms is True

    See Parser class for usage examples.

    """
    def __init__(self, collapseTerms=True, collapseAtoms=False):
        super().__init__()
        self.collapseTerms = bool(collapseTerms)
        self.collapseAtoms = bool(collapseAtoms)

    def visit_number(self, node, children):
        return str(node.value) if self.collapseTerms else int(node.value)

    def visit_args(self, node, children):
        return children

    def visit_subterm(self, node, children):
        predicate, *args = children
        if self.collapseTerms:
            return (predicate + '(' + ','.join(*args) + ')') if args else predicate
        else:
            return Term(predicate, list(args[0])) if args else predicate

    def visit_term(self, node, children):
        predicate, *args = children
        if self.collapseAtoms:
            return (predicate + '(' + ','.join(*args) + ')') if args else predicate
        else:
            return Term(predicate, list(args[0]) if args else [])

    def visit_terms(self, node, children):
        return TermSet(children)

    @staticmethod
    def grammar():
        def ident():      return ap.RegExMatch(r'[a-z][a-zA-Z0-9_]*')
        def number():     return ap.RegExMatch(r'-?[0-9]+')
        def litteral():   return [ap.RegExMatch(r'"[^"]*"'), number]
        def subterm():    return [(ident, ap.Optional("(", args, ")")), litteral]
        def args():       return subterm, ap.ZeroOrMore(',', subterm)
        def term():       return [(ident, ap.Optional("(", args, ")")), litteral]
        def terms():      return ap.ZeroOrMore(term)
        return terms


class AtomVisitor(ap.PTNodeVisitor):
    """Implement both the grammar and the way to handle it, dedicated to the
    parsing of ASP like string to produce TermSet instances.

    This is a simpler version than CollapsableAtomVisitor, which implement
    special use cases, notably partial cut of parsed atoms.

    """

    def visit_number(self, node, children):
        return int(node.value)

    def visit_args(self, node, children):
        return children

    def visit_term(self, node, children):
        predicate, *args = children
        return Term(predicate, list(args[0])) if args else predicate

    def visit_terms(self, node, children):
        return TermSet(children)

    @staticmethod
    def grammar():
        def ident():      return ap.RegExMatch(r'[a-z][a-zA-Z0-9_]*')
        def number():     return ap.RegExMatch(r'-?[0-9]+')
        def litteral():   return [ap.RegExMatch(r'"[^"]*"'), number]
        def args():       return term, ap.ZeroOrMore(',', term)
        def term():       return [(ident, ap.Optional("(", args, ")")), litteral]
        def terms():      return ap.ZeroOrMore(term)
        return terms


class Parser:
    def __init__(self, collapseTerms=True, collapseAtoms=False, callback=None):
        """
        collapseTerms: function terms in predicate arguments are collapsed into strings
        collapseAtoms: atoms (predicate plus terms) are collapsed into strings
                       requires that collapseTerms is True

        examples:

            >>> Parser(True, False).parse_terms('a(b,c(d))')
            TermSet({Term('a',['b','c(d)'])})

            >>> Parser(True, True).parse_terms('a(b,c(d))')
            TermSet({'a(b,c(d))'})

            >>> Parser(False, False).parse_terms('a(b,c(d))')
            TermSet({Term('a',['b',Term('c',['d'])])})

            >>> Parser(False, True).parse_terms('a(b,c(d))')  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ...
            ValueError

        """
        self.collapseTerms = bool(collapseTerms)
        self.collapseAtoms = bool(collapseAtoms)
        if not self.collapseTerms and not self.collapseAtoms:  # optimized case
            self.atom_visitor = AtomVisitor()
        else:
            self.atom_visitor = CollapsableAtomVisitor(bool(collapseTerms), bool(collapseAtoms))
        self.grammar = self.atom_visitor.grammar()
        self.callback = callback
        if self.collapseAtoms and not self.collapseTerms:
            raise ValueError("if atoms are collapsed, terms must"
                             " also be collapsed!")


    def parse_terms(self, string:str) -> TermSet:
        """Return the TermSet computed from given valid ASP-compliant string"""
        parse_tree = ap.ParserPython(self.grammar).parse(string)
        if parse_tree:
            return ap.visit_parse_tree(parse_tree, self.atom_visitor)
        else:
            return TermSet()

    # alias
    parse = parse_terms


    def parse_clasp_output(self, output:iter or str, *, yield_stats:bool=False,
                           yield_info:bool=False):
        """Yield pairs (payload type, payload) where type is 'info', 'statistics'
        or 'answer' and payload the raw information.

        output -- iterable of lines or full clasp output to parse
        yield_stats -- yields final statistics as a mapping {field: value}
                       under type 'statistics'
        yield_info  -- yields first lines not related to first answer
                       under type 'info' as a tuple of lines

        In any case, tuple ('answer', termset) will be returned
        with termset a TermSet instance containing all atoms in the found model.

        See test/test_parsing.py for examples.

        """
        REGEX_ANSWER_HEADER = re.compile(r"Answer: [0-9]+")
        output = iter(output.splitlines() if isinstance(output, str) else output)
        modes = iter(('info', 'answers', 'statistics'))
        mode = next(modes)

        # all lines until meeting "Answer: 1" belongs to the info lines
        info = []
        line = ''
        line = next(output)
        while not REGEX_ANSWER_HEADER.fullmatch(line):
            info.append(line)
            line = next(output)
        if yield_info:
            yield 'info', tuple(info)

        # first answer begins
        while True:
            if REGEX_ANSWER_HEADER.fullmatch(line):
                next_line = next(output)
                answer = self.parse_terms(next_line)
                yield 'answer', answer
            if not line.strip():  # empty line: statistics are beginning
                if not yield_stats: break  # stats are the last part of the output
                stats = {}
                for line in output:
                    sep = line.find(':')
                    key, value = line[:sep], line[sep+1:]
                    stats[key.strip()] = value.strip()
                yield 'statistics', stats
            line = next(output)



class Lexer:
    tokens = (
        'STRING',
        'IDENT',
        'MIDENT',
        'NUM',
        'LP',
        'RP',
        'COMMA',
        'SPACE',
    )

    # Tokens
    t_STRING = r'"((\\")|[^"])*"'
    t_IDENT = r'[a-zA-Z_][a-zA-Z0-9_]*'
    t_MIDENT = r'-[a-zA-Z_][a-zA-Z0-9_]*'
    t_NUM = r'-?[0-9]+'
    t_LP = r'\('
    t_RP = r'\)'
    t_COMMA = r','
    t_SPACE = r'[ \t\.]+'

    def __init__(self):
        self.lexer = lex.lex(object=self, optimize=OPTIMIZE, lextab='asp_py_lextab')

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self, t):
        print("Illegal character " + str(t.value[0]))
        t.lexer.skip(1)


class OldParser:
    start = 'answerset'

    def __init__(self, collapseTerms=True, collapseAtoms=False, callback=None):
        """
        collapseTerms: function terms in predicate arguments are collapsed into strings
        collapseAtoms: atoms (predicate plus terms) are collapsed into strings
                       requires that collapseTerms is True

        examples:


        """
        self.accu = TermSet()
        self.lexer = Lexer()
        self.tokens = self.lexer.tokens
        self.collapseTerms = collapseTerms
        self.collapseAtoms = collapseAtoms
        self.callback = callback

        if collapseAtoms and not collapseTerms:
            raise ValueError("if atoms are collapsed, functions must"
                             " also be collapsed!")

        self.parser = yacc.yacc(module=self, optimize=OPTIMIZE, tabmodule='asp_py_parsetab')

    def p_answerset(self, t):
        """answerset : atom SPACE answerset
                   | atom
        """
        self.accu.add(t[1])

    def p_atom(self, t):
        """atom : IDENT LP terms RP
                | IDENT
                | MIDENT LP terms RP
                | MIDENT
        """
        if self.collapseAtoms:
            if len(t) == 2:
                t[0] = str(t[1])
            elif len(t) == 5:
                t[0] = "%s(%s)" % ( t[1], ",".join(map(str, t[3])) )
        else:
            if len(t) == 2:
                t[0] = Term(t[1])
            elif len(t) == 5:
                t[0] = Term(t[1], t[3])

    def p_terms(self, t):
        """terms : term COMMA terms
                 | term
        """
        if len(t) == 2:
            t[0] = [t[1]]
        else:
            t[0] = [t[1]] + t[3]

    def p_term(self, t):
        """term : IDENT LP terms RP
                | STRING
                | IDENT
                | NUM
        """
        if self.collapseTerms:
            if len(t) == 2:
                t[0] = t[1]
            else:
                t[0] = t[1] + "(" + ",".join(t[3]) + ")"
        else:
            if len(t) == 2:
                if re.match(r'-?[0-9]+', t[1]) != None:
                    t[0] = int(t[1])
                else:
                    t[0] = t[1]
            else:
                t[0] = Term(t[1], t[3])

    def p_error(self, t):
        print("Syntax error at "+str(t))
        print (''.join(map(lambda x: "  %s:%s\n    %s" % (x[1], x[2], x[4][0]),
                           inspect.stack())))

    def parse(self, line):
        self.accu = TermSet()
        line = line.strip()

        if len(line) > 0:
            self.parser.parse(line, lexer=self.lexer.lexer)

        if self.callback:
            self.callback(self.accu)

        return self.accu

    parse_terms = parse


def filter_empty_str(l):
    return [x for x in l if x != '']
