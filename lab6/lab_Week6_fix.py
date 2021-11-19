'''
Authors: Henry Thompson, Bharat Ram Ambati
Date: 2014-10-01
Copyright: This work is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License
(http://creativecommons.org/licenses/by-nc/4.0/): You may re-use,
redistribute, or modify this work for non-commercial purposes provided
you retain attribution to any previous author(s).
'''
# fix buggy NLTK 3 :-(
# different fixes for different versions :-((((
import re, sys
import nltk
from nltk.grammar import _ARROW_RE, _PROBABILITY_RE, _DISJUNCTION_RE, Production
from nltk.draw import CFGEditor
from nltk.probability import ImmutableProbabilisticMixIn
ARROW = u'\u2192'
TOKEN = u'([\\w ]|\\\\((x[0-9a-f][0-9a-f])|(u[0-9a-f][0-9a-f][0-9a-f][0-9a-f])))+'
CFGEditor.ARROW = ARROW
CFGEditor._TOKEN_RE=re.compile(u"->|u?'"+TOKEN+u"'|u?\""+TOKEN+u"\"|\\w+|("+ARROW+u")")
CFGEditor._PRODUCTION_RE=re.compile(u"(^\s*\w+\s*)" +
                                    u"(->|("+ARROW+"))\s*" +
                                    u"((u?'"+TOKEN+"'|u?\""+TOKEN+"\"|''|\"\"|\w+|\|)\s*)*$")
nltk.grammar._TERMINAL_RE = re.compile(u'( u?"[^"]+" | u?\'[^\']+\' ) \s*', re.VERBOSE)
nltk.grammar._ARROR_RE = re.compile(u'\s* (->|'+ARROW+') \s*', re.VERBOSE)

from nltk.grammar import _TERMINAL_RE

if sys.version_info[0]>2 or sys.version_info[1]>6:
    from nltk.grammar import PCFG, CFG, ProbabilisticProduction as FixPP
    parse_grammar=CFG.fromstring
    parse_pgrammar=PCFG.fromstring
    from nltk import InsideChartParser
    def nbest_parse(self,tokens,n=None):
        parses=self.parse(tokens)
        if n is None:
            return [parse for parse in parses]
        else:
            return [parses.next() for i in range(n)]
    InsideChartParser.nbest_parse=nbest_parse
else:
    from nltk.grammar import WeightedGrammar as PCFG, WeightedProduction as FixPP
    from nltk import parse_cfg, parse_pcfg
    parse_grammar=parse_cfg
    parse_pgrammar=parse_pcfg

def fix_parse_production(line, nonterm_parser, probabilistic=False):
    """
    Parse a grammar rule, given as a string, and return
    a list of productions.
    """
    pos = 0

    # Parse the left-hand side.
    lhs, pos = nonterm_parser(line, pos)

    # Skip over the arrow.
    m = _ARROW_RE.match(line, pos)
    if not m: raise ValueError('Expected an arrow')
    pos = m.end()

    # Parse the right hand side.
    probabilities = [0.0]
    rhsides = [[]]
    while pos < len(line):
        # Probability.
        m = _PROBABILITY_RE.match(line, pos)
        if probabilistic and m:
            pos = m.end()
            probabilities[-1] = float(m.group(1)[1:-1])
            if probabilities[-1] > 1.0:
                raise ValueError('Production probability %f, '
                                 'should not be greater than 1.0' %
                                 (probabilities[-1],))

        # String -- add terminal.
        elif (line[pos] in "\'\"" or line[pos:pos+2] in ('u"',"u'")):
            m = _TERMINAL_RE.match(line, pos)
            if not m: raise ValueError('Unterminated string')
            rhsides[-1].append(eval(m.group(1)))
            pos = m.end()

        # Vertical bar -- start new rhside.
        elif line[pos] == '|':
            m = _DISJUNCTION_RE.match(line, pos)
            probabilities.append(0.0)
            rhsides.append([])
            pos = m.end()

        # Anything else -- nonterminal.
        else:
            nonterm, pos = nonterm_parser(line, pos)
            rhsides[-1].append(nonterm)

    if probabilistic:
        return [FixPP(lhs, rhs, prob=probability)
                for (rhs, probability) in zip(rhsides, probabilities)]
    else:
        return [Production(lhs, rhs) for rhs in rhsides]

if sys.version_info[0]>2 or sys.version_info[1]>6:
    nltk.grammar._read_production=fix_parse_production
else:
    nltk.grammar.parse_production=fix_parse_production

class CostedProduction(FixPP):
    """
    A probabilistic context free grammar production using costs.
    A PCFG ``ProbabilisticProduction`` is essentially just a ``Production`` that
    has an associated probability, which represents how likely it is that
    this production will be used.  In particular, the probability of a
    ``ProbabilisticProduction`` records the likelihood that its right-hand side is
    the correct instantiation for any given occurrence of its left-hand side.

    :see: ``Production``
    """
    def __init__(self, lhs, rhs, cost):
        """
        Construct a new ``ProbabilisticProduction``.

        :param lhs: The left-hand side of the new ``ProbabilisticProduction``.
        :type lhs: Nonterminal
        :param rhs: The right-hand side of the new ``ProbabilisticProduction``.
        :type rhs: sequence(Nonterminal and terminal)
        :param prob: Probability parameters of the new ``ProbabilisticProduction``.
        """
        ImmutableProbabilisticMixIn.__init__(self, logprob=-cost)
        Production.__init__(self, lhs, rhs)

    def __str__(self):
        return Production.__unicode__(self) + \
            (' [0.0]' if (self.logprob() == 0.0) else ' [%g]' % -self.logprob())

    def __repr__(self):
        return '%s'%str(self)

    def cost(self):
        return 0.0 if self.logprob() == 0.0 else -self.logprob()
