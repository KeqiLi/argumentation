# Carneades Argument Evaluation Structure
#
# Copyright (C) 2014 Ewan Klein
# Author: Ewan Klein <ewan@inf.ed.ac.uk>
# Based on: https://hackage.haskell.org/package/CarneadesDSL
#
# For license information, see LICENSE

"""
==========
 Overview
==========

Propositions
============

First, let's create some propositions using the :class:`PropLiteral`
constructor. All propositions are atomic, that is, either positive or
negative literals.

>>> kill = PropLiteral('kill')
>>> kill.polarity
True

>>> intent = PropLiteral('intent')
>>> murder = PropLiteral('murder')
>>> witness1 = PropLiteral('witness1')
>>> unreliable1 = PropLiteral('unreliable1')
>>> witness2 = PropLiteral('witness2')
>>> unreliable2 = PropLiteral('unreliable2')

The :meth:`negate` method allows us to introduce negated propositions.

>>> neg_intent = intent.negate()
>>> print(neg_intent)
-intent
>>> neg_intent.polarity
False
>>> neg_intent == intent
False
>>> neg_intent.negate() == intent
True

Arguments
=========

Arguments are built with the :class:`Argument` constructor. They are required
to have a conclusion, and may also have premises and exceptions.

>>> arg1 = Argument(murder, premises={kill, intent})
>>> arg2 = Argument(intent, premises={witness1}, exceptions={unreliable1})
>>> arg3 = Argument(neg_intent, premises={witness2}, exceptions={unreliable2})
>>> print(arg1)
[intent, kill], ~[] => murder

In order to organise the dependencies between the conclusion of an argument
and its premises and exceptions, we model them using a directed graph called
an :class:`ArgumentSet`. Notice that the premise of one argument (e.g., the
``intent`` premise of ``arg1``) can be the conclusion of another argument (i.e.,
``arg2``)).

>>> argset = ArgumentSet()
>>> argset.add_argument(arg1, arg_id='arg1')
>>> argset.add_argument(arg2, arg_id='arg2')
>>> argset.add_argument(arg3, arg_id='arg3')

There is a :func:`draw` method which allows us to view the resulting graph.

>>> argset.draw() # doctest: +SKIP

Proof Standards
===============

In evaluating the relative value of arguments for a particular conclusion
``p``, we need to determine what standard of *proof* is required to establish
``p``. The notion of proof used here is not formal proof in a logical
system. Instead, it tries to capture how substantial the arguments are
in favour of, or against, a particular conclusion.

The :class:`ProofStandard` constructor is initialised with a list of
``(proposition, name-of-proof-standard)`` pairs. The default proof standard,
viz., ``'scintilla'``, is the weakest level.  Different
propositions can be assigned different proof standards that they need
to attain.

>>> ps = ProofStandard([(intent, "beyond_reasonable_doubt")],
... default='scintilla')

Carneades Argument Evaluation Structure
=======================================

The core of the argumentation model is a data structure plus set of
rules for evaluating arguments; this is called a Carneades Argument
Evaluation Structure (CAES). A CAES consists of a set of arguments,
an audience (or jury), and a method for determining whether propositions
satisfy the relevant proof standards.

The role of the audience is modeled as an :class:`Audience`, consisting
of a set of assumed propositions, and an assignment of weights to
arguments.

>>> assumptions = {kill, witness1, witness2, unreliable2}
>>> weights = {'arg1': 0.8, 'arg2': 0.3, 'arg3': 0.8}
>>> audience = Audience(assumptions, weights)

Once an audience has been defined, we can use it to initialise a
:class:`CAES`, together with instances of :class:`ArgumentSet` and
:class:`ProofStandard`:

>>> caes = CAES(argset, audience, ps)
>>> caes.get_all_arguments()
[intent, kill], ~[] => murder
[witness1], ~[unreliable1] => intent
[witness2], ~[unreliable2] => -intent

The :meth:`get_arguments` method returns the list of arguments in an
:class:`ArgumentSet` which support a given proposition.

A proposition is said to be *acceptable* in a CAES if it meets its required
proof standard. The process of checking whether a proposition meets its proof
standard requires another notion: namely, whether the arguments that support
it are *applicable*. An argument ``arg`` is applicable if and only if all its
premises either belong to the audience's assumptions or are acceptable;
moreover, the exceptions of ``arg`` must not belong to the assumptions or be
acceptable. For example, `arg2`, which supports the conclusion `intent`, is
acceptable since `witness1` is an assumption, while the exception
`unreliable1` is neither an assumption nor acceptable.

>>> arg_for_intent = argset.get_arguments(intent)[0]
>>> print(arg_for_intent)
[witness1], ~[unreliable1] => intent
>>> caes.applicable(arg_for_intent)
True

>>> caes.acceptable(intent)
False

Although there is an argument (``arg3``) for `-intent`, it is not applicable,
since the exception `unreliable2` does belong to the audience's assumptions.

>>> any(caes.applicable(arg) for arg in argset.get_arguments(neg_intent))
False

This in turn has the consequence that `-intent` is not acceptable.

>>> caes.acceptable(neg_intent)
False

Despite the fact that the argument `arg2` for `murder` is applicable,
the conclusion `murder` is not acceptable, since

>>> caes.acceptable(murder)
False
>>> caes.acceptable(murder.negate())
False


"""


from collections import namedtuple, defaultdict
import logging
import os
import sys

from igraph import Graph, plot

# fix to ensure that package is loaded properly on system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from carneades.tracecalls import TraceCalls


# LOGLEVEL = logging.DEBUG
# Uncomment the following line to raise the logging level and thereby turn off
# debug messages
# LOGLEVEL = logging.INFO


class PropLiteral(object):
    """
    Proposition literals have most of the properties of ordinary strings,
    except that the negation method is Boolean; i.e.

    >>> a = PropLiteral('a')
    >>> a.negate().negate() == a
    True
    """
    def __init__(self, string, polarity=True):
        """
        Propositions are either positive or negative atoms.
        """
        self.polarity = polarity
        self._string = string


    def negate(self):
        """
        Negation of a proposition.

        We create a copy of the current proposition and flip its polarity.
        """
        polarity = (not self.polarity)
        return PropLiteral(self._string, polarity=polarity)


    def __str__(self):
        """
        Override ``__str__()`` so that negation is realised as a prefix on the
        string.
        """
        if self.polarity:
            return self._string
        return "-" + self._string

    def __hash__(self):
        return self._string.__hash__()

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__str__() == other.__str__()
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.__str__() < other.__str__()





class Argument(object):
    """
    An argument consists of a conclusion, a set of premises and a set of
    exceptions (both of which can be empty).

    Although arguments should have identifiers (`arg_id`), it is preferable
    to specify these when calling the :meth:`add_argument` method of
    :class:`ArgumentSet`.
    """
    def __init__(self, conclusion, premises=set(), exceptions=set()):
        """
        :param conclusion: The conclusion of the argument.
        :type conclusion: :class:`PropLiteral`
        :param premises: The premises of the argument.
        :type premises: set(:class:`PropLiteral`)
        :param exceptions: The exceptions of the argument
        :type exceptions: set(:class:`PropLiteral`)
        """

        self.conclusion = conclusion
        self.premises = premises
        self.exceptions = exceptions
        self.arg_id = None


    def __str__(self):
        """
        Define print string for arguments.

        We follow similar conventions to those used by the CarneadesDSL
        Haskell implementation.

        Premises and exceptions are sorted to facilitate doctest comparison.
        """
        if len(self.premises) == 0:
            prems = "[]"
        else:
            prems = sorted(self.premises)
        if len(self.exceptions) == 0:
            excepts = "[]"
        else:
            excepts = sorted(self.exceptions)
        return "{}, ~{} => {}".format(prems, excepts, self.conclusion)


class ArgumentSet(object):
    """
    An ``ArgumentSet`` is modeled as a dependency graph where vertices represent
    the components of an argument. A vertex corresponding to the conclusion
    of an argument *A* will **depend on** the premises and exceptions in *A*.

    The graph is built using the `igraph <http://igraph.org/>`_ library. This
    allows *attributes* to be associated with both vertices and edges.
    Attributes are represented as Python dictionaries where the key (which
    must be a string) is the name of the attribute and the value is the
    attribute itself. For more details, see the
    `igraph tutorial\
    <http://igraph.org/python/doc/tutorial/tutorial.html#setting-and-retrieving-attributes>`_.
    """
    def __init__(self):
        self.graph = Graph()
        self.graph.to_directed()
        self.arg_count = 1
        self.arguments = []

    def propset(self):
        """
        The set of :class:`PropLiteral`\ s represented by the vertices in
        the graph.

        Retrieving this set relies on the fact that :meth:`add_proposition`
        sets a value for the ``prop`` attribute in vertices created when a
        new proposition is added to the graph.
        """
        g = self.graph
        props = set()
        try:
            props = {p for p in g.vs['prop']}
        except KeyError:
            pass
        return props

    def add_proposition(self, proposition):
        """
        Add a proposition to a graph if it is not already present as a vertex.

        :param proposition: The proposition to be added to the graph.
        :type proposition: :class:`PropLiteral`
        :return: The graph vertex corresponding to the proposition.
        :rtype: :class:`Graph.Vertex`
        :raises TypeError: if the input is not a :class:`PropLiteral`.
        """
        if isinstance(proposition, PropLiteral):
            if proposition in self.propset():
                logging.debug("Proposition '{}' is already in graph".\
                              format(proposition))
            else:
                # add the proposition as a vertex attribute, recovered via the
                # key 'prop'
                self.graph.add_vertex(prop=proposition)
                logging.debug("Added proposition '{}' to graph".\
                              format(proposition))
            return self.graph.vs.select(prop=proposition)[0]

        else:
            raise TypeError('Input {} should be PropLiteral'.\
                            format(proposition))

    def add_argument(self, argument, arg_id=None):
        """
        Add an argument to the graph.

        :parameter argument: The argument to be added to the graph.
        :type argument: :class:`Argument`
        :parameter arg_id: The ID of the argument
        :type arg_id: str or None
        """
        g = self.graph
        if arg_id is not None:
            argument.arg_id = arg_id
        else:
            argument.arg_id = 'arg{}'.format(self.arg_count)
        self.arg_count += 1
        self.arguments.append(argument)

        # add the arg_id as a vertex attribute, recovered via the 'arg' key
        self.graph.add_vertex(arg=argument.arg_id)
        arg_v = g.vs.select(arg=argument.arg_id)[0]

        # add proposition vertices to the graph
        conclusion_v = self.add_proposition(argument.conclusion)
        self.add_proposition(argument.conclusion.negate())
        premise_vs =\
            [self.add_proposition(prop) for prop in sorted(argument.premises)]
        exception_vs =\
            [self.add_proposition(prop) for prop in sorted(argument.exceptions)]
        target_vs = premise_vs + exception_vs

        # add new edges to the graph
        edge_to_arg = [(conclusion_v.index, arg_v.index)]
        edges_from_arg = [(arg_v.index, target.index) for target in target_vs]
        g.add_edges(edge_to_arg + edges_from_arg)


    def get_arguments(self, proposition):
        """
        Find the arguments for a proposition in an *ArgumentSet*.

        :param proposition: The proposition to be checked.
        :type proposition: :class:`PropLiteral`
        :return: A list of the arguments pro the proposition
        :rtype: list(:class:`Argument`)

        :raises ValueError: if the input :class:`PropLiteral` isn't present\
        in the graph.
        """
        g = self.graph

        # index of vertex associated with the proposition
        vs = g.vs.select(prop=proposition)

        try:
            conc_v_index = vs[0].index
            # IDs of vertices reachable in one hop from the proposition's vertex
            target_IDs = [e.target for e in g.es.select(_source=conc_v_index)]

            # the vertices indexed by target_IDs
            out_vs = [g.vs[i] for i in target_IDs]

            arg_IDs = [v['arg'] for v in out_vs]
            args = [arg for arg in self.arguments if arg.arg_id in arg_IDs]
            return args
        except IndexError:
            raise ValueError("Proposition '{}' is not in the current graph".\
                             format(proposition))





    def draw(self, debug=False):
        """
        Visualise an :class:`ArgumentSet` as a labeled graph.

        :parameter debug: If :class:`True`, add the vertex index to the label.
        """
        g = self.graph

        # labels for nodes that are classed as propositions
        labels = g.vs['prop']

        # insert the labels for nodes that are classed as arguments
        for i in range(len(labels)):
            if g.vs['arg'][i] is not None:
                labels[i] = g.vs['arg'][i]

        if debug:
            d_labels = []
            for (i, label) in enumerate(labels):
                d_labels.append("{}\nv{}".format(label, g.vs[i].index))

            labels = d_labels

        g.vs['label'] = labels


        roots = [i for i in range(len(g.vs)) if g.indegree()[i] == 0]
        ALL = 3 # from igraph
        layout = g.layout_reingold_tilford(mode=ALL, root=roots)

        plot_style = {}
        plot_style['vertex_color'] = \
            ['lightblue' if x is None else 'pink' for x in g.vs['arg']]
        plot_style['vertex_size'] = 60
        plot_style['vertex_shape'] = \
            ['circle' if x is None else 'rect' for x in g.vs['arg']]
        plot_style['margin'] = 40
        plot_style['layout'] = layout
        #plot(g, **plot_style)



    def write_to_graphviz(self, fname=None):
        g = self.graph
        result = "digraph G{ \n"

        for vertex in g.vs:
            arg_label = vertex.attributes()['arg']
            prop_label = vertex.attributes()['prop']

            if arg_label:
                dot_str = (arg_label +
                           ' [color="black", fillcolor="pink", width=.75, '
                           'shape=box, style="filled"]; \n')

            elif prop_label:
                dot_str = ('"{}"'.format(prop_label) +
                           ' [color="black", fillcolor="lightblue", '
                           'fixedsize=true, width=1  shape="circle", '
                           'style="filled"]; \n')
            result += dot_str

        for edge in g.es:
            source_label = g.vs[edge.source]['prop'] if\
                g.vs[edge.source]['prop'] else g.vs[edge.source]['arg']
            target_label = g.vs[edge.target]['prop'] if\
                g.vs[edge.target]['prop'] else g.vs[edge.target]['arg']
            result += '"{}" -> "{}"'.format(source_label, target_label)
            dot_str = " ; \n"
            result += dot_str

        result += "}"

        if fname is None:
            fname = 'graph.dot'
        with open(fname, 'w') as f:
            print(result, file=f)


class ProofStandard(object):
    """
    Each proposition in a CAES is associated with a proof standard.

    A proof standard is initialised by supplying a (possibly empty) list of
    pairs, each consisting of a proposition and the name of a proof standard.

    >>> intent = PropLiteral('intent')
    >>> ps = ProofStandard([(intent, "beyond_reasonable_doubt")])

    Possible values for proof standards: `"scintilla"`, `"preponderance"`,
    `"clear_and_convincing"`, `"beyond_reasonable_doubt"`, and
    `"dialectical_validity"`.
    """
    def __init__(self, propstandards, default='scintilla'):
        """
        :param propstandards: the proof standard associated with\
        each proposition under consideration.
        :type propstandards: list(tuple(:class:`PropLiteral`, str))
        """
        self.proof_standards = ["scintilla", "preponderance",
                                "clear_and_convincing",
                                "beyond_reasonable_doubt",
                                "dialectical_validity"]
        self.default = default
        self.config = defaultdict(lambda: self.default)
        self._set_standard(propstandards)

    def _set_standard(self, propstandards):
        for (prop, standard) in propstandards:
            if standard not in self.proof_standards:
                raise ValueError("{} is not a valid proof standard".\
                                 format(standard))
            self.config[prop] = standard


    def get_proofstandard(self, proposition):
        """
        Determine the proof standard associated with a proposition.

        :param proposition: The proposition to be checked.
        :type proposition: :class:`PropLiteral`
        """
        return self.config[proposition]


Audience = namedtuple('Audience', ['assumptions', 'weight'])
"""
An audience has assumptions about which premises hold and also
assigns weights to arguments.

:param assumptions: The assumptions held by the audience
:type assumptions: set(:class:`PropLiteral`)

:param weights: An mapping from :class:`Argument`\ s to weights.
:type weights: dict
"""

class CAES(object):
    """
    A class that represents a Carneades Argument Evaluation Structure (CAES).

    """
    def __init__(self, argset, audience, proofstandard, alpha=0.4, beta=0.3,
                 gamma=0.2):
        """
        :parameter argset: the argument set used in the CAES
        :type argset: :class:`ArgSet`

        :parameter audience: the audience for the CAES
        :type audience: :class:`Audience`

        :parameter proofstandard: the proof standards used in the CAES
        :type proofstandard: :class:`ProofStandard`

        :parameter alpha: threshold of strength of argument required for a\
        proposition to reach the proof standards "clear and convincing" and\
        "beyond reasonable doubt".

        :type alpha: float in interval [0, 1]


        :parameter beta: difference required between strength of\
        argument *pro* a proposition vs strength of argument *con*\
        to reach the proof standard "clear and convincing".

        :type beta: float in interval [0, 1]

        :parameter gamma: threshold of strength of a *con* argument required\
        for a proposition to reach the proof standard "beyond reasonable\
        doubt".

        :type gamma: float in interval [0, 1]
        """
        self.argset = argset
        self.assumptions = audience.assumptions
        self.weight = audience.weight
        self.standard = proofstandard
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def get_all_arguments(self):
        """
        Show all arguments in the :class:`ArgSet` of the CAES.
        """
        for arg in self.argset.arguments:
            print(arg)

    @TraceCalls()
    def applicable(self, argument):
        """
        An argument is *applicable* in a CAES if it needs to be taken into
        account when evaluating the CAES.

        :parameter argument: The argument whose applicablility is being\
        determined.

        :type argument: :class:`Argument`
        :rtype: bool
        """
        _acceptable = lambda p: self.acceptable(p)
        return self._applicable(argument, _acceptable)

    def _applicable(self, argument, _acceptable):
        """
        :parameter argument: The argument whose applicablility is being
        determined.

        :type argument: :class:`Argument`

        :parameter _acceptable: The function which determines the
        acceptability of a proposition in the CAES.

        :type _acceptable: LambdaType
        :rtype: bool
        """
        logging.debug('Checking applicability of {}...'.format(argument.arg_id))
        logging.debug('Current assumptions: {}'.format(self.assumptions))
        logging.debug('Current premises: {}'.format(argument.premises))
        b1 = all(p in self.assumptions or \
                 (p.negate() not in self.assumptions and \
                  _acceptable(p)) for p in argument.premises)

        if argument.exceptions:
            logging.debug('Current exception: {}'.format(argument.exceptions))
        b2 = all(e not in self.assumptions and \
                 (e.negate() in self.assumptions or \
                  not _acceptable(e)) for e in argument.exceptions)

        return b1 and b2


    @TraceCalls()
    def acceptable(self, proposition):
        """
        A conclusion is *acceptable* in a CAES if it can be arrived at under
        the relevant proof standards, given the beliefs of the audience.

        :param proposition: The conclusion whose acceptability is to be\
        determined.

        :type proposition: :class:`PropLiteral`

        :rtype: bool
        """

        standard = self.standard.get_proofstandard(proposition)
        logging.debug("Checking whether proposition '{}'"
                      "meets proof standard '{}'.".\
                      format(proposition, standard))
        return self.meets_proof_standard(proposition, standard)

    @TraceCalls()
    def meets_proof_standard(self, proposition, standard):
        """
        Determine whether a proposition meets a given proof standard.

        :param proposition: The proposition which should meet the relevant\
        proof standard.

        :type proposition: :class:`PropLiteral`

        :parameter standard: a specific level of proof;\
        see :class:`ProofStandard` for admissible values

        :type standard: str
        :rtype: bool

        """
        arguments = self.argset.get_arguments(proposition)

        result = False

        if standard == 'scintilla':
            result = any(arg for arg in arguments if self.applicable(arg))
        elif standard == 'preponderance':
            result = self.max_weight_pro(proposition) > \
                self.max_weight_con(proposition)
        elif standard == 'clear_and_convincing':
            mwp = self.max_weight_pro(proposition)
            mwc = self.max_weight_con(proposition)
            exceeds_alpha = mwp > self.alpha
            diff_exceeds_gamma = (mwp - mwc) > self.gamma
            logging.debug("max weight pro '{}' is {}".format(proposition, mwp))
            logging.debug("max weight con '{}' is {}".format(proposition, mwc))
            logging.debug("max weight pro '{}' >  alpha '{}': {}".\
                          format(mwp, self.alpha, exceeds_alpha))
            logging.debug("diff between pro and con = {} > gamma: {}".\
                          format(mwp-mwc, diff_exceeds_gamma))

            result = (mwp > self.alpha) and (mwp - mwc > self.gamma)
        elif standard == 'beyond_reasonable_doubt':
            result = self.meets_proof_standard(proposition,
                                               'clear_and_convincing') \
                and \
                self.max_weight_con(proposition) < self.gamma
        elif standard == 'dialectical_validity':
            result = self.meets_proof_standard(proposition,
                                               'scintilla')\
                and \
                not(self.meets_proof_standard(proposition.negate(), 'scintilla'))
        return result

    def weight_of(self, argument):
        """
        Retrieve the weight associated by the CAES audience with an argument.

        :parameter argument: The argument whose weight is being determined.
        :type argument: :class:`Argument`
        :return: The weight of the argument.
        :rtype: float in interval [0, 1]
        """
        arg_id = argument.arg_id
        try:
            return self.weight[arg_id]
        except KeyError:
            raise ValueError("No weight assigned to argument '{}'.".\
                             format(arg_id))


    def max_weight_applicable(self, arguments):
        """
        Retrieve the weight of the strongest applicable argument in a list
        of arguments.

        :parameter arguments: The arguments whose weight is being compared.
        :type arguments: list(:class:`Argument`)
        :return: The maximum of the weights of the arguments.
        :rtype: float in interval [0, 1]
        """
        arg_ids = [arg.arg_id for arg in arguments]

        applicable_args = [arg for arg in arguments if self.applicable(arg)]
        if len(applicable_args) == 0:
            logging.debug('No applicable arguments in {}'.format(arg_ids))
            return 0.0

        applic_arg_ids = [arg.arg_id for arg in applicable_args]
        logging.debug('Checking applicability and weights of {}'.\
                      format(applic_arg_ids))
        weights = [self.weight_of(argument) for argument in applicable_args]
        logging.debug('Weights of {} are {}'.format(applic_arg_ids, weights))
        return max(weights)

    def max_weight_pro(self, proposition):
        """
        The maximum of the weights pro the proposition.

        :param proposition: The conclusion whose acceptability is to be\
        determined.

        :type proposition: :class:`PropLiteral`
        :rtype: float in interval [0, 1]
        """
        args = self.argset.get_arguments(proposition)
        return self.max_weight_applicable(args)

    def max_weight_con(self, proposition):
        """
        The maximum of the weights con the proposition.

        :param proposition: The conclusion whose acceptability is to be\
        determined.

        :type proposition: :class:`PropLiteral`
        :rtype: float in interval [0, 1]
        """
        con = proposition.negate()
        args = self.argset.get_arguments(con)
        return self.max_weight_applicable(args)




def arg_demo():
    """
    Demo of how to initialise and call methods of a CAES.
    """
    kill = PropLiteral('kill')
    intent = PropLiteral('intent')
    neg_intent = intent.negate()
    murder = PropLiteral('murder')
    witness1 = PropLiteral('witness1')
    unreliable1 = PropLiteral('unreliable1')
    witness2 = PropLiteral('witness2')
    unreliable2 = PropLiteral('unreliable2')

    ps = ProofStandard([(intent, "beyond_reasonable_doubt")])

    arg1 = Argument(murder, premises={kill, intent})
    arg2 = Argument(intent, premises={witness1}, exceptions={unreliable1})
    arg3 = Argument(neg_intent, premises={witness2}, exceptions={unreliable2})

    argset = ArgumentSet()
    argset.add_argument(arg1)
    argset.add_argument(arg2)
    argset.add_argument(arg3)
    argset.draw()
    argset.write_to_graphviz()

    assumptions = {kill, witness1, witness2, unreliable2}
    weights = {'arg1': 0.8, 'arg2': 0.3, 'arg3': 0.8}
    audience = Audience(assumptions, weights)
    caes = CAES(argset, audience, ps)
    caes.acceptable(murder)
    caes.acceptable(murder.negate())


class Reader():
    """
    This Reader class can read in a txt file and build up a caes
    argumentation system to judge.

    >>> reader = Reader()

    """
    def __init__(self, data=None):
        self.data = data
        self.propositions = {}
        self.argset = ArgumentSet()
        self.prosecution = {}
        self.defence = {}
        self.weights = {}
        self.assumptions = {}
        self.proof_standards = {}
        self.target = ''
        self.read_status = ''
        self.bop = 'P'
        self.keywords = ['PROPOSITIONS','ARGUMENTS','ASSUMPTIONS',\
                         'WEIGHTS', 'PROOF STANDARDS', 'TARGETS']

    def burden(self, data):
        """
        This is the function of the class to use.
        :parameter data: The data that contains the information from the txt file.
        :type data: string
        """
        lines = data.read().splitlines()
        for line in lines:
            # Ignore the comments
            line = line.split('#')[0]

            # Against exploitation/mistakes
            count_key = 0
            for key in self.keywords:
                if key in line:
                    count_key = count_key + 1
            if count_key > 1:
                raise IndexError('Too many (expected 1) keywords in the line '+line)

            # Define what it is reading next and finish the processing
            self.check_status(line)
        

        ps = ProofStandard([])

        # Audience construction
        audience = Audience(self.assumptions.values(), self.weights)

        # Build up CAES system 
        round = 0
        currentArgs = {}
        accept = False
        dialogue = ''

        change = 1
        while change == 1:
            change = 0
            dialogue = dialogue + '\nDialogue State {}:'.format(round) +'\n'
            dialogue = dialogue + 'Current arguments: '+'\n'
            for arg in currentArgs:
                dialogue = dialogue + str(currentArgs[arg]) + '\n'
            dialogue = dialogue + 'Accept {}? ==> {}'.format(self.target, accept)+'\n'
            dialogue = dialogue + 'Burden of Proof: ' + self.bop+'\n'

            if round == 0:
                argno, effectivearg = self.find_pro_arg(self.target)
                if effectivearg != 0:
                    self.argset.add_argument(effectivearg)
                    currentArgs[argno] = effectivearg
                    change = 1

            elif self.bop == 'P':
                argno, effectivearg = self.prosec(currentArgs, audience, ps)
                if effectivearg != 0:
                    currentArgs[argno] = effectivearg
                    change = 1

            elif self.bop == 'D':
                argno, effectivearg = self.defencing(currentArgs, audience, ps)
                if effectivearg != 0:
                    currentArgs[argno] = effectivearg
                    change = 1

            tempps = []
            for key in self.proof_standards:
                if PropLiteral(key, True) in self.argset.propset()\
                   and PropLiteral(key, False)in self.argset.propset():
                   tempps.append((self.propositions[key], self.proof_standards[key]))

            ps = ProofStandard(tempps)
            accept = CAES(self.argset, audience, ps).acceptable(self.target)
            if accept:
                self.bop = 'D'
            else:
                self.bop = 'P'

            round = round + 1

        print(dialogue)
        args = self.argset.get_arguments(PropLiteral('selfdef', True))
        for arg in args:
            print(arg)

    def find_pro_arg(self, conclusion):
        for arg in self.prosecution:
            if conclusion == self.prosecution[arg].conclusion:
                return arg, self.prosecution[arg]
        return 0, 0

    def find_def_arg(self, conclusion):
        for arg in self.defence:
            if conclusion == self.defence[arg].conclusion:
                return arg, self.defence[arg]
        return 0, 0

    def defencing(self, currentArgs, audience, ps):
        # Self proving
        for arg in currentArgs:
            if arg in self.defence.keys():
                caes = CAES(self.argset, audience, ps)
                if not caes.acceptable(currentArgs[arg].conclusion):
                    for prem in currentArgs[arg].premises:
                        if not caes.acceptable(prem):
                            argno, effectivearg = self.find_def_arg(prem)
                            if effectivearg != 0 and argno not in currentArgs:
                                self.argset.add_argument(effectivearg)
                                return argno, effectivearg
        # rebutting
        for arg in currentArgs:
            if arg in self.prosecution.keys():
                argno, effectivearg = self.find_def_arg(currentArgs[arg].conclusion.negate())
                if effectivearg != 0 and argno not in currentArgs:
                    self.argset.add_argument(effectivearg)
                    return argno, effectivearg

        # exception
        for arg in currentArgs:
            if arg in self.prosecution.keys():
                for exception in currentArgs[arg].exceptions:
                    argno, effectivearg = self.find_def_arg(exception)
                    if effectivearg != 0 and argno not in currentArgs:
                        self.argset.add_argument(effectivearg)
                        return argno, effectivearg
        return 0, 0

    def prosec(self, currentArgs, audience, ps):
        # Self proving
        for arg in currentArgs:
            if arg in self.prosecution.keys():
                caes = CAES(self.argset, audience, ps)
                if not caes.acceptable(currentArgs[arg].conclusion):
                    for prem in currentArgs[arg].premises:
                        if not caes.acceptable(prem):
                            argno, effectivearg = self.find_pro_arg(prem)
                            if effectivearg != 0 and argno not in currentArgs:
                                self.argset.add_argument(effectivearg)
                                return argno, effectivearg

        # rebutting
        for arg in currentArgs:
            if arg in self.defence.keys():
                argno, effectivearg = self.find_pro_arg(currentArgs[arg].conclusion.negate())
                if effectivearg != 0 and argno not in currentArgs:
                    self.argset.add_argument(effectivearg)
                    return argno, effectivearg

        # exception
        for arg in currentArgs:
            if arg in self.defence.keys():
                for exception in currentArgs[arg].exceptions:
                    argno, effectivearg = self.find_pro_arg(exception)
                    if effectivearg != 0 and argno not in currentArgs:
                        self.argset.add_argument(effectivearg)
                        return argno, effectivearg
        return 0, 0

    def check_status(self, line):
        """
        This function defines the status that the reader is reading for
        a specific line.
        >>> reader = Reader()
        >>> reader.check_status('ARGUMENTS')
        >>> reader.read_status
        'args'
        >>> reader.check_status('PROPOSITIONS')
        >>> reader.read_status
        'props'

        :parameter line: A line from the text file
        :type line: String

        """
        if 'PROPOSITIONS' in line:
            self.read_status = 'props'
        elif 'ARGUMENTS' in line:
            self.read_status = 'args'
        elif 'ASSUMPTIONS' in line:
            self.read_status = 'assums'
        elif 'WEIGHTS' in line:
            self.read_status = 'wts'
        elif 'PROOF STANDARDS' in line:
            self.read_status = 'ps'
        elif 'TARGET' in line :
            self.read_status = 'tars'
        else:
            # Content line, not header attrubutes.
            self.assign(line)

    def assign(self, line):
        """
        According to the reading status, assign each line to 
        the specific function.
        >>> reader = Reader()
        >>> reader.check_status('PROPOSITIONS')
        >>> reader.assign('-happy')
        >>> reader.propositions['-happy']
        -happy

        :parameter line: A line from the text file
        :type line: String

        """
        length = len(line.split())
        if length > 0:
            # Propositions
            if self.read_status == 'props': 
                if length == 1:
                    self.add_prop(line.split()[0])
                else:
                    raise IndexError('Too many tokens in line '+line)
            # Arguments
            elif self.read_status == 'args': 
                self.add_arg(''.join(line.split()))
            # Assumptions
            elif self.read_status == 'assums':
                if length == 1:
                    self.add_assum(line.split()[0])
                else:
                    raise IndexError('Too many tokens in line '+line)
            # Weights
            elif self.read_status == 'wts': 
                self.add_weight(''.join(line.split()))
            # Proof Standards
            elif self.read_status == 'ps': 
                self.add_proof_standard(''.join(line.split()))
            # Targets
            elif self.read_status == 'tars': 
                if length == 1:
                    self.add_target(line.split()[0])
                else:
                    raise IndexError('Too many tokens in line '+line)
            # No reading status -> Error
            elif self.read_status == '':
                raise TypeError('Missing keywords at the beginning of file.')

    def add_prop(self, proposition):
        """
        Add one proposition one time and stored in a dictionary
        Also deal with error that double negations appear
        >>> reader = Reader()
        >>> reader.add_prop('-happy')
        >>> reader.propositions['-happy']
        -happy

        :parameter line: A line from the text file
        :type line: String

        :raises ValueError: if the input has multiple hypens.
        """
        if proposition[0] != '-':
            self.propositions[proposition] = PropLiteral(proposition)
        elif proposition[0:2] == '--':
            raise ValueError('Please do not use multiple negation when \
                              trying to intialize propositions.')
        else:
            self.propositions[proposition] = PropLiteral(proposition[1:], False)

    def add_arg(self, arg):
        """
        Argument reading

        >>> reader = Reader()
        >>> reader.add_prop('a')
        >>> reader.add_prop('b')
        >>> reader.add_prop('c')
        >>> reader.add_arg('arg:[a], ~[b] => c')
        >>> print(reader.argset.get_arguments(PropLiteral('c'))[0])
        [a], ~[b] => c

        :parameter arg: A line from the text file
        :type arg: String
        """

        # Splitting the argument line
        arg = ''.join(arg.split())
        argname, statement = arg.split(':')
        stance = argname[0]
        argname = argname[1:]
        rest, conclusion = statement.split('=>')
        if conclusion not in self.propositions.keys():
            self.add_prop(conclusion)
        conclusion = self.propositions[conclusion]

        if '~' in rest:
            # There is exception
            premises, exceptions = rest.split('],~[')

            # Deal with premises
            premises = premises[1:]
            premises = premises.split(',')
            [self.add_prop(key) for key in premises]
            premises = [self.propositions[key] for key in premises]

            # Deal with exceptions
            exceptions = exceptions[:-1]
            exceptions = exceptions.split(',')
            [self.add_prop(key) for key in exceptions]     
            exceptions = [self.propositions[key] for key in exceptions]
        else:
            # There is no exception
            premises = rest[1:-1]
            premises = premises.split(',')
            [self.add_prop(key) for key in premises]
            premises = [self.propositions[key] for key in premises]
            exceptions = set()

        # Forming argument and add it to the argset
        argument = Argument(conclusion, premises = premises, exceptions=exceptions)
        if stance == 'P':
            self.prosecution[argname] = argument
        elif stance == 'D':
            self.defence[argname] = argument

    def add_assum(self, assumption):
        """
        Add an assumption
        Takes in one assumption
        Deal with error
        >>> reader = Reader()
        >>> reader.add_prop('a')
        >>> reader.add_assum('a')
        >>> reader.assumptions['a']
        a

        :parameter assumption: an assumption of the caes file
        :type assumption: String

        :raises IndexError: if the assumption is not in the graph.
        """
        
        if assumption in self.propositions.keys():
            self.assumptions[assumption] = self.propositions[assumption]
        else:
            raise IndexError('Assumption not in the graph')

    def add_target(self, target):
        """
        Add a target
        If such target is not in the graph already, add it to the graph
        >>> reader = Reader()
        >>> reader.add_target('a')
        >>> reader.targets['a']
        a

        :parameter target: a proposition that the caes system wants to prove
        :type assumption: String
        """

        if target in self.propositions.keys():
            self.target = self.propositions[target]
        else:
            self.add_prop(target)
            self.target = self.propositions[target]

    def add_weight(self, line): 
        """
        Add a weight to the dictionary

        >>> reader = Reader()
        >>> reader.add_weight('arg:6')
        >>> reader.weights['arg']
        6.0

        :parameter line: a line that contains weight of an argument
        :type line: String
        """
        line = ''.join(line.split())
        argname, weight = line.split(':')
        self.weights[argname] = float(weight)

    def add_proof_standard(self, line):
        """
        Add a proof standard to dictionary ps

        >>> reader = Reader()
        >>> reader.add_proof_standard('a: dialectical_validity')
        >>> reader.proof_standards['a']
        'dialectical_validity'

        :parameter line: a line that contains a proof standard
        :type line: String
        """
        line = ''.join(line.split())
        prop, ps = line.split(':')
        self.proof_standards[prop] = ps;



if __name__ == '__main__':
    #Instruction for the user
    debug_mode = input("Would you like to enable the debug mode? (y/n)\n")
    if debug_mode == 'n':
        LOGLEVEL = logging.INFO
    elif debug_mode == 'y':
        LOGLEVEL = logging.DEBUG
    else:
        raise ValueError('Please enter only y or n')
    logging.basicConfig(format='%(levelname)s: %(message)s', level=LOGLEVEL)

    file_name = input('Please enter the filename to test burden of proof (eg. caesfile.txt):\n')
    caes_data = open(file_name)
    reader = Reader(1)
    reader.burden(caes_data)

