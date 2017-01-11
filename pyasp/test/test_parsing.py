

import pytest

from pyasp import parsing
from pyasp.parsing import Parser, OldParser
from pyasp.term import Term, TermSet


def test_simple_case():
    parsed = Parser().parse_clasp_output(OUTCLASP_SIMPLE.splitlines(),
                                         yield_stats=True, yield_info=True)
    models = []
    for type, payload in parsed:
        assert type in ('statistics', 'info', 'answer')
        if type == 'statistics':
            stats = payload
        elif type == 'answer':
            models.append(payload)
        else:
            assert type == 'info'
            info = payload
    assert len(models) == 2
    assert info == ('clasp version 3.2.0', 'Reading from stdin', 'Solving...')
    assert stats == {'Models': '2', 'Calls': '1', 'CPU Time': '0.000s',
                     'Time': '0.001s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)'}


def test_parse_termset_default():
    string = 'a(b,10) c(d("a",d_d),"v,v",c) d(-2,0)'
    expected = TermSet((
        Term('a', ['b', '10']),
        Term('c', ['d("a",d_d)', '"v,v"', 'c']),
        Term('d', ['-2', '0']),
    ))
    assert Parser().parse_terms(string) == expected


def test_parse_termset():
    string = 'a(b,10) c(d("a",d_d),"v,v",c) d(-2,0)'
    expectation = {
        (True, False): TermSet((
            Term('a', ['b', '10']),
            Term('c', ['d("a",d_d)', '"v,v"', 'c']),
            Term('d', ['-2', '0']),
        )),
        (False, False): TermSet((
            Term('a', ['b', 10]),
            Term('c', [Term('d', ['"a"', 'd_d']), '"v,v"', 'c']),
            Term('d', [-2, 0]),
        )),
        (True, True): TermSet((
            'a(b,10)',
            'c(d("a",d_d),"v,v",c)',
            'd(-2,0)',
        )),
        # False True is expected to raise an Error (see dedicated function)
    }
    for parser_mode, expected in expectation.items():
        print(*parser_mode)
        assert OldParser(*parser_mode).parse_terms(string) == expected
        assert    Parser(*parser_mode).parse_terms(string) == expected

def test_parse_termset_impossible():
    with pytest.raises(ValueError) as e_info:
        OldParser(False, True)
    with pytest.raises(ValueError) as e_info:
        Parser(False, True)


def test_complex_atoms():
    parsed = Parser().parse_clasp_output(OUTCLASP_COMPLEX_ATOMS.splitlines())
    type, model = next(parsed)
    assert next(parsed, None) is None, "there is only one model"
    assert type == 'answer', "the model is an answer"
    assert len(model) == 3, "only 3 atoms in it"
    atoms = {atom.predicate: atom for atom in model}
    assert set(atoms.keys()) == {'a', 'b', 'c'}
    # TODO: make the term parser


def test_optimization():
    parsed = Parser().parse_clasp_output(OUTCLASP_OPTIMIZATION.splitlines(), yield_stats=True)
    expected_stats = {
        'CPU Time': '0.130s',
        'Calls': '1',
        'Models': '1',
        'Optimization': '190',
        'Optimum': 'yes',
        'Threads': '4        (Winner: 2)',
        'Time': '0.051s (Solving: 0.03s 1st Model: 0.00s Unsat: 0.03s)'
    }
    for type, model in parsed:
        if type == 'statistics':
            assert model == expected_stats


def test_unsat():
    parsed = Parser().parse_clasp_output(OUTCLASP_UNSATISFIABLE.splitlines())
    assert next(parsed, None) is None



OUTCLASP_COMPLEX_ATOMS = """clasp version 3.2.0
Reading from stdin
Solving...
Answer: 1
a(b(1,"a")) b(42,12) c("42,12,a(23)",c)
SATISFIABLE

Models       : 1
Calls        : 1
Time         : 0.001s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)
CPU Time     : 0.000s
"""

OUTCLASP_SIMPLE = """clasp version 3.2.0
Reading from stdin
Solving...
Answer: 1
a
Answer: 2
b
SATISFIABLE

Models       : 2
Calls        : 1
Time         : 0.001s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)
CPU Time     : 0.000s
"""

OUTCLASP_OPTIMIZATION = """clasp version 3.2.0
Reading from l.lp
Solving...
Answer: 1
set(2,9) set(2,10) score(20)
Optimization: 190
OPTIMUM FOUND

Models       : 1
  Optimum    : yes
Optimization : 190
Calls        : 1
Time         : 0.051s (Solving: 0.03s 1st Model: 0.00s Unsat: 0.03s)
CPU Time     : 0.130s
Threads      : 4        (Winner: 2)
"""

OUTCLASP_UNSATISFIABLE = """clasp version 3.2.0
Reading from l.lp
Solving...
UNSATISFIABLE

Models       : 0
Calls        : 1
Time         : 0.001s (Solving: 0.00s 1st Model: 0.00s Unsat: 0.00s)
CPU Time     : 0.000s
"""