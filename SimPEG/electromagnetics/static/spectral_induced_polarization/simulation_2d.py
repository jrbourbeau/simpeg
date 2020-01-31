import numpy as np
from scipy.special import kn
import properties

from .... import props
from .... import maps
from ....utils import sdiag

from ..resistivity.fields_2d import (
    Fields_ky, Fields_ky_CC, Fields_ky_N
)
from ..induced_polarization.simulation_2d import BaseIPSimulation_2D
from ..induced_polarization import Problem2D_N as BaseProblem2D_N
from ..induced_polarization import Problem2D_CC as BaseProblem2D_CC
from .survey import Survey
from .simulation import BaseSIPSimulation


class BaseSIPSimulation_2D(BaseIPSimulation_2D, BaseSIPSimulation):


    eta, etaMap, etaDeriv = props.Invertible(
        "Electrical Chargeability (V/V)"
    )

    tau, tauMap, tauDeriv = props.Invertible(
        "Time constant (s)",
        default=0.1
    )

    taui, tauiMap, tauiDeriv = props.Invertible(
        "Inverse of time constant (1/s)"
    )

    props.Reciprocal(tau, taui)

    c, cMap, cDeriv = props.Invertible(
        "Frequency dependency",
        default=1.
    )

    survey = properties.Instance(
        "an SIP survey object", Survey, required=True
    )

    # surveyPair = Survey
    fieldsPair = Fields_ky
    _f = None
    _Jmatrix = None
    actinds = None
    actMap = None

    _eta_store = None
    _taui_store = None
    _c_store = None
    _pred = None

    @property
    def etaDeriv_store(self):
        if getattr(self, '_etaDeriv_store', None) is None:
            self._etaDeriv_store = self.etaDeriv
        return self._etaDeriv_store

    @property
    def tauiDeriv_store(self):
        if getattr(self, '_tauiDeriv_store', None) is None:
            self._tauiDeriv_store = self.tauiDeriv
        return self._tauiDeriv_store

    @property
    def tauDeriv_store(self):
        if getattr(self, '_tauDeriv_store', None) is None:
            self._tauDeriv_store = self.tauDeriv
        return self._tauDeriv_store

    @property
    def cDeriv_store(self):
        if getattr(self, '_cDeriv_store', None) is None:
            self._cDeriv_store = self.cDeriv
        return self._cDeriv_store

    def getJ(self, m, f=None):
        """
            Generate Full sensitivity matrix
        """

        if self.verbose:
            print(">> Compute Sensitivity matrix")

        if self._Jmatrix is not None:
            return self._Jmatrix
        else:

            if f is None:
                f = self.fields(m)

            Jt = np.zeros(
                (self.actMap.nP, int(self.survey.nD/self.survey.times.size)),
                order='F'
            )
            istrt = int(0)
            iend = int(0)

            # Assume y=0.
            dky = np.diff(self.kys)
            dky = np.r_[dky[0], dky]
            y = 0.
            for src in self.survey.source_list:
                for rx in src.receiver_list:
                    iend = istrt + rx.nD
                    Jtv_temp1 = np.zeros(
                        (self.actinds.sum(), rx.nD), dtype=float
                    )
                    Jtv_temp0 = np.zeros(
                        (self.actinds.sum(), rx.nD), dtype=float
                    )
                    # TODO: this loop is pretty slow .. (Parellize)
                    for iky in range(self.nky):
                        u_src = f[src, self._solutionType, iky]
                        ky = self.kys[iky]

                        # wrt f, need possibility wrt m
                        P = rx.getP(self.mesh, rx.projGLoc(f)).toarray()

                        ATinvdf_duT = self.Ainv[iky] * (P.T)

                        dA_dmT = self.getADeriv(ky, u_src, ATinvdf_duT,
                                                adjoint=True)
                        Jtv_temp1 = 1./np.pi*(-dA_dmT)
                        # Trapezoidal intergration
                        if iky == 0:
                            # First assigment
                            if rx.nD == 1:
                                Jt[:, istrt] += Jtv_temp1*dky[iky]*np.cos(ky*y)
                            else:
                                Jt[:, istrt:iend] += (
                                    Jtv_temp1*dky[iky]*np.cos(ky*y)
                                )
                        else:
                            if rx.nD == 1:
                                Jt[:, istrt] += (
                                    Jtv_temp1*dky[iky]/2.*np.cos(ky*y)
                                )
                                Jt[:, istrt] += (
                                    Jtv_temp0*dky[iky]/2.*np.cos(ky*y)
                                )
                            else:
                                Jt[:, istrt:iend] += (
                                    Jtv_temp1*dky[iky]/2.*np.cos(ky*y)
                                )
                                Jt[:, istrt:iend] += (
                                    Jtv_temp0*dky[iky]/2.*np.cos(ky*y)
                                )
                        Jtv_temp0 = Jtv_temp1.copy()
                    istrt += rx.nD

            self._Jmatrix = Jt.T
            # delete fields after computing sensitivity
            del f
            if self._f is not None:
                self._f = []
            # clean all factorization
            if self.Ainv[0] is not None:
                for i in range(self.nky):
                    self.Ainv[i].clean()
            return self._Jmatrix

    def forward(self, m, f=None):

        self.model = m

        self._eta_store = self.eta
        self._taui_store = self.taui
        self._c_store = self.c

        J = self.getJ(m, f=f)

        ntime = len(self.survey.times)
        Jv = []
        self.model = m
        for tind in range(ntime):
            Jv.append(
                J.dot(
                    self.actMap.P.T*self.get_peta(self.survey.times[tind]))
                )
        return self.sign * np.hstack(Jv)

    def dpred(self, m, f=None):
        """
            Predicted data.

            .. math::
                d_\\text{pred} = Pf(m)
        """
        if f is None:
            f = self.fields(m)
        return self._pred
        # return self.forward(m, f=f)

    def Jvec(self, m, v, f=None):

        self.model = m
        J = self.getJ(m, f=f)
        ntime = len(self.survey.times)
        Jv = []

        for tind in range(ntime):

            t = self.survey.times[tind]
            v0 = self.PetaEtaDeriv(t, v)
            v1 = self.PetaTauiDeriv(t, v)
            v2 = self.PetaCDeriv(t, v)
            PTv = self.actMap.P.T*(v0+v1+v2)
            Jv.append(J.dot(PTv))

        return self.sign * np.hstack(Jv)

    def Jtvec(self, m, v, f=None):

        self.model = m
        J = self.getJ(m, f=f)

        ntime = len(self.survey.times)
        Jtvec = np.zeros(m.size)
        v = v.reshape((int(self.survey.nD/ntime), ntime), order="F")

        for tind in range(ntime):
            t = self.survey.times[tind]
            Jtv = self.actMap.P*J.T.dot(v[:, tind])
            Jtvec += (
                self.PetaEtaDeriv(t, Jtv, adjoint=True) +
                self.PetaTauiDeriv(t, Jtv, adjoint=True) +
                self.PetaCDeriv(t, Jtv, adjoint=True)
            )

        return self.sign * Jtvec

    def getJtJdiag(self, m):
        """
        Compute JtJ using adjoint problem. Still we never form
        JtJ
        """
        ntime = len(self.survey.times)
        JtJdiag = np.zeros_like(m)
        J = self.getJ(m, f=None)
        for tind in range(ntime):
            t = self.survey.times[tind]
            Jtv = self.actMap.P*J.T
            JtJdiag += (
                (self.PetaEtaDeriv(t, Jtv, adjoint=True)**2).sum(axis=1) +
                (self.PetaTauiDeriv(t, Jtv, adjoint=True)**2).sum(axis=1) +
                (self.PetaCDeriv(t, Jtv, adjoint=True)**2).sum(axis=1)
            )
        return JtJdiag

    @property
    def MfRhoDerivMat(self):
        """
        Derivative of MfRho with respect to the model
        """
        if getattr(self, '_MfRhoDerivMat', None) is None:
            drho_dlogrho = sdiag(self.rho)*self.actMap.P
            self._MfRhoDerivMat = self.mesh.getFaceInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nF)) * drho_dlogrho
        return self._MfRhoDerivMat

    def MfRhoIDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MfRhoI` with respect to the model.
        """
        dMfRhoI_dI = -self.MfRhoI**2

        if self.storeInnerProduct:
            if adjoint:
                return (
                    self.MfRhoDerivMat.T * (
                        sdiag(u) * (dMfRhoI_dI.T * v)
                    )
                )
            else:
                return dMfRhoI_dI * (sdiag(u) * (self.MfRhoDerivMat*v))
        else:
            drho_dlogrho = sdiag(self.rho)*self.actMap.P
            dMf_drho = self.mesh.getFaceInnerProductDeriv(self.rho)(u)
            if adjoint:
                return drho_dlogrho.T * (dMf_drho.T * (dMfRhoI_dI.T*v))
            else:
                return dMfRhoI_dI * (dMf_drho * (drho_dlogrho*v))

    @property
    def MeSigmaDerivMat(self):
        """
        Derivative of MeSigma with respect to the model
        """
        if getattr(self, '_MeSigmaDerivMat', None) is None:
            dsigma_dlogsigma = sdiag(self.sigma)*self.actMap.P
            self._MeSigmaDerivMat = self.mesh.getEdgeInnerProductDeriv(
                np.ones(self.mesh.nC)
            )(np.ones(self.mesh.nE)) * dsigma_dlogsigma
        return self._MeSigmaDerivMat

    # TODO: This should take a vector
    def MeSigmaDeriv(self, u, v, adjoint=False):
        """
        Derivative of MeSigma with respect to the model times a vector (u)
        """

        if self.storeInnerProduct:
            if adjoint:
                return self.MeSigmaDerivMat.T * (sdiag(u)*v)
            else:
                return sdiag(u)*(self.MeSigmaDerivMat * v)
        else:
            dsigma_dlogsigma = sdiag(self.sigma)*self.actMap.P
            if adjoint:
                return (
                    dsigma_dlogsigma.T * (
                        self.mesh.getEdgeInnerProductDeriv(self.sigma)(u).T * v
                    )
                )
            else:
                return (
                    self.mesh.getEdgeInnerProductDeriv(self.sigma)(u) *
                    (dsigma_dlogsigma * v)
                )

    @property
    def MnSigmaDerivMat(self):
        """
            Derivative of MnSigma with respect to the model
        """
        if getattr(self, '_MnSigmaDerivMat', None) is None:
            sigma = self.sigma
            vol = self.mesh.vol
            dsigma_dlogsigma = sdiag(sigma)*self.actMap.P
            self._MnSigmaDerivMat = (
                self.mesh.aveN2CC.T * sdiag(vol) * dsigma_dlogsigma
                )
        return self._MnSigmaDerivMat

    def MnSigmaDeriv(self, u, v, adjoint=False):
        """
            Derivative of MnSigma with respect to the model times a vector (u)
        """
        if self.storeInnerProduct:
            if adjoint:
                return self.MnSigmaDerivMat.T * (sdiag(u)*v)
            else:
                return u*(self.MnSigmaDerivMat * v)
        else:
            sigma = self.sigma
            vol = self.mesh.vol
            dsigma_dlogsigma = sdiag(sigma)*self.actMap.P
            if adjoint:
                return dsigma_dlogsigma.T * (vol * (self.mesh.aveN2CC * (u*v)))
            else:
                dsig_dm_v = dsigma_dlogsigma * v
                return (
                    u * (self.mesh.aveN2CC.T * (vol * dsig_dm_v))
                )

    @property
    def MccRhoiDerivMat(self):
        """
            Derivative of MccRho with respect to the model
        """
        if getattr(self, '_MccRhoiDerivMat', None) is None:
            rho = self.rho
            vol = self.mesh.vol
            drho_dlogrho = sdiag(rho)*self.actMap.P
            self._MccRhoiDerivMat = (
                sdiag(vol*(-1./rho**2))*drho_dlogrho
            )
        return self._MccRhoiDerivMat

    def MccRhoiDeriv(self, u, v, adjoint=False):
        """
            Derivative of :code:`MccRhoi` with respect to the model.
        """
        if len(self.rho.shape) > 1:
            if self.rho.shape[1] > self.mesh.dim:
                raise NotImplementedError(
                    "Full anisotropy is not implemented for MccRhoiDeriv."
                )
        if self.storeInnerProduct:
            if adjoint:
                return self.MccRhoiDerivMat.T * (sdiag(u) * v)
            else:
                return sdiag(u) * (self.MccRhoiDerivMat * v)
        else:
            vol = self.mesh.vol
            rho = self.rho
            drho_dlogrho = sdiag(rho)*self.actMap.P
            if adjoint:
                return drho_dlogrho.T * (u*vol*(-1./rho**2) * v)
            else:
                return (u*vol*(-1./rho**2))*(drho_dlogrho * v)

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = [
            '_etaDeriv_store', '_tauiDeriv_store', '_cDeriv_store',
            '_tauDeriv_store'
        ]
        return toDelete

class Problem2D_CC(BaseSIPSimulation_2D, BaseProblem2D_CC):
    """
    2.5D cell centered Spectral IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'HJ'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_CC
    sign = 1.
    bc_type = "Mixed"

    def __init__(self, mesh, **kwargs):
        BaseSIPSimulation_2D.__init__(self, mesh, **kwargs)
        if self.actinds is None:
            print("You did not put Active indices")
            print("So, set actMap = IdentityMap(mesh)")
            self.actinds = np.ones(mesh.nC, dtype=bool)

        self.actMap = maps.InjectActiveCells(mesh, self.actinds, 0.)


class Problem2D_N(BaseSIPSimulation_2D, BaseProblem2D_N):
    """
    2.5D nodal Spectral IP problem
    """

    _solutionType = 'phiSolution'
    _formulation = 'EB'  # CC potentials means J is on faces
    fieldsPair = Fields_ky_N
    sign = -1.

    def __init__(self, mesh, **kwargs):
        BaseSIPSimulation_2D.__init__(self, mesh, **kwargs)
        # self.setBC()
        if self.actinds is None:
            print("You did not put Active indices")
            print("So, set actMap = IdentityMap(mesh)")
            self.actinds = np.ones(mesh.nC, dtype=bool)

        self.actMap = maps.InjectActiveCells(mesh, self.actinds, 0.)
