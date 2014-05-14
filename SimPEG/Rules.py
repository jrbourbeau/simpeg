import Utils, numpy as np

class InversionRule(object):
    """InversionRule"""

    debug    = False    #: Print debugging information

    current = None     #: This hold

    def __init__(self, **kwargs):
        Utils.setKwargs(self, **kwargs)

    @property
    def inversion(self):
        """This is the inversion of the InversionRule instance."""
        return getattr(self,'_inversion',None)
    @inversion.setter
    def inversion(self, i):
        if getattr(self,'_inversion',None) is not None:
            print 'Warning: InversionRule %s has switched to a new inversion.' % self.__name__
        self._inversion = i

    @property
    def objFunc(self): return self.inversion.objFunc
    @property
    def opt(self): return self.inversion.opt
    @property
    def reg(self): return self.inversion.objFunc.reg
    @property
    def survey(self): return self.inversion.objFunc.survey
    @property
    def prob(self): return self.inversion.objFunc.prob

    def initialize(self):
        pass

    def endIter(self):
        pass

    def finish(self):
        pass

class RuleList(object):

    rList = None   #: The list of Rules

    def __init__(self, *rules, **kwargs):
        self.rList = []
        for r in rules:
            assert isinstance(r, InversionRule), 'All rules must be InversionRules not %s' % r.__name__
            self.rList.append(r)
        Utils.setKwargs(self, **kwargs)

    @property
    def debug(self):
        return getattr(self, '_debug', False)
    @debug.setter
    def debug(self, value):
        for r in self.rList:
            r.debug = value
        self._debug = value

    @property
    def inversion(self):
        """This is the inversion of the InversionRule instance."""
        return getattr(self,'_inversion',None)
    @inversion.setter
    def inversion(self, i):
        if self.inversion is i: return
        if getattr(self,'_inversion',None) is not None:
            print 'Warning: %s has switched to a new inversion.' % self.__name__
        for r in self.rList:
            r.inversion = i
        self._inversion = i

    def call(self, ruleType):
        if self.rList is None:
            if self.debug: 'RuleList is None, no rules to call!'
            return

        rules = ['initialize', 'endIter', 'finish']
        assert ruleType in rules, 'Rule type must be in ["%s"]' % '", "'.join(rules)
        for r in self.rList:
            getattr(r, ruleType)()


class BetaEstimate_ByEig(InversionRule):
    """BetaEstimate"""

    beta0 = None       #: The initial Beta (regularization parameter)
    beta0_ratio = 0.1  #: estimateBeta0 is used with this ratio

    def initialize(self):
        """
            The initial beta is calculated by comparing the estimated
            eigenvalues of JtJ and WtW.

            To estimate the eigenvector of **A**, we will use one iteration
            of the *Power Method*:

            .. math::

                \mathbf{x_1 = A x_0}

            Given this (very course) approximation of the eigenvector,
            we can use the *Rayleigh quotient* to approximate the largest eigenvalue.

            .. math::

                \lambda_0 = \\frac{\mathbf{x^\\top A x}}{\mathbf{x^\\top x}}

            We will approximate the largest eigenvalue for both JtJ and WtW, and
            use some ratio of the quotient to estimate beta0.

            .. math::

                \\beta_0 = \gamma \\frac{\mathbf{x^\\top J^\\top J x}}{\mathbf{x^\\top W^\\top W x}}

            :rtype: float
            :return: beta0
        """

        if self.debug: print 'Calculating the beta0 parameter.'

        m = self.objFunc.m_current
        u = self.objFunc.u_current or self.prob.fields(m)

        x0 = np.random.rand(*m.shape)
        t = x0.dot(self.objFunc.dataObj2Deriv(m,x0,u=u))
        b = x0.dot(self.reg.modelObj2Deriv(m, v=x0))
        self.beta0 = self.beta0_ratio*(t/b)

        self.objFunc.beta = self.beta0


class BetaSchedule(InversionRule):
    """BetaSchedule"""

    coolingFactor = 2.
    coolingRate = 3

    def endIter(self):
        if self.opt.iter > 0 and self.opt.iter % self.coolingRate == 0:
            if self.debug: print 'BetaSchedule is cooling Beta. Iteration: %d' % self.opt.iter
            self.objFunc.beta /= self.coolingFactor


# class UpdateReferenceModel(Parameter):

#     mref0 = None

#     def nextIter(self):
#         mref = getattr(self, 'm_prev', None)
#         if mref is None:
#             if self.debug: print 'UpdateReferenceModel is using mref0'
#             mref = self.mref0
#         self.m_prev = self.objFunc.m_current
#         return mref
