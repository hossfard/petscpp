#include "Ksp.h"
#include "petscksp.h"
#include "petscpc.h"
#include <algorithm>
#include "Petscpp.h"

using namespace Petscpp;


Ksp::
Ksp(Type type, PPreconditioner::Type pcType)
  : ksp_(nullptr)
{
  KSPCreate(PETSC_COMM_WORLD, &ksp_);
  PC pc;
  KSPGetPC(ksp_, &pc);
  setType(type);
  preconditioner().setType(pcType);
}


Ksp::
~Ksp(){
  KSPDestroy(&ksp_);
}


KSP const&
Ksp::
petscKsp() const{
  return ksp_;
}


KSP&
Ksp::
petscKsp(){
  return ksp_;
}


std::string
Petscpp::
toString(Ksp::Type type){
  switch (type){
  case (Ksp::Richardson):
    return KSPRICHARDSON;
  case (Ksp::Chebyshev):
    return KSPCHEBYSHEV;
  case (Ksp::Cg):
    return KSPCG;
  case (Ksp::Groppcg):
    return KSPGROPPCG;
  case (Ksp::Pipecg):
    return KSPPIPECG;
  case (Ksp::Cgne):
    return KSPCGNE;
  case (Ksp::Nash):
    return KSPNASH;
  case (Ksp::Stcg):
    return KSPSTCG;
  case (Ksp::Gltr):
    return KSPGLTR;
  case (Ksp::Gmres):
    return KSPGMRES;
  case (Ksp::Fgmres):
  case (Ksp::Lgmres):
    return KSPLGMRES;
  case (Ksp::Dgmres):
    return KSPDGMRES;
  case (Ksp::Pgmres):
    return KSPPGMRES;
  case (Ksp::Tcqmr):
    return KSPTCQMR;
  case (Ksp::Ibcgs):
    return KSPIBCGS;
  case (Ksp::Fbcgs):
    return KSPFBCGS;
  case (Ksp::Fbcgsr):
    return KSPFBCGSR;
  case (Ksp::Bcgsl):
    return KSPBCGSL;
  case (Ksp::Cgs):
    return KSPCGS;
  case (Ksp::Tfqmr):
    return KSPTFQMR;
  case (Ksp::Cr):
    return KSPCR;
  case (Ksp::Pipecr):
    return KSPPIPECR;
  case (Ksp::Lsqr):
    return KSPLSQR;
  case (Ksp::Preonly):
    return KSPPREONLY;
  case (Ksp::Qcg):
    return KSPQCG;
  case (Ksp::Bicg):
    return KSPBICG;
  case (Ksp::Minres):
    return KSPMINRES;
  case (Ksp::Symmlq):
    return KSPSYMMLQ;
  case (Ksp::Lcd):
    return KSPLCD;
  case (Ksp::Python):
    return KSPPYTHON;
  case (Ksp::Gcr):
    return KSPGCR;
  case (Ksp::Specest):
    return KSPSPECEST;
  default:
    return "";
  }
}


void
Ksp::
setType(Ksp::Type type){
  std::string str = toString(type);
  if (str != "")
    KSPSetType(ksp_, str.c_str());
}


Ksp::Type
Ksp::
type() const{
  KSPType t;
  KSPGetType(ksp_, &t);

  if (strcmp(t,KSPRICHARDSON) == 0)
    return Ksp::Richardson;
  else if (strcmp(t,KSPCHEBYSHEV) == 0)
    return Ksp::Chebyshev;
  else if (strcmp(t,KSPCG) == 0)
    return Ksp::Cg;
  else if (strcmp(t,KSPGROPPCG) == 0)
    return Ksp::Groppcg;
  else if (strcmp(t,KSPPIPECG) == 0)
    return Ksp::Pipecg;
  else if (strcmp(t,KSPCGNE) == 0)
    return Ksp::Cgne;
  else if (strcmp(t,KSPNASH) == 0)
    return Ksp::Nash;
  else if (strcmp(t,KSPSTCG) == 0)
    return Ksp::Stcg;
  else if (strcmp(t,KSPGLTR) == 0)
    return Ksp::Gltr;
  else if (strcmp(t,KSPGMRES) == 0)
    return Ksp::Gmres;
  else if (strcmp(t, KSPFGMRES) == 0)
    return Ksp::Fgmres;
  else if (strcmp(t, KSPLGMRES) == 0)
    return Ksp::Lgmres;
  else if (strcmp(t, KSPDGMRES) == 0)
    return Ksp::Dgmres;
  else if (strcmp(t, KSPPGMRES) == 0)
    return Ksp::Pgmres;
  else if (strcmp(t, KSPTCQMR) == 0)
    return Ksp::Tcqmr;
  else if (strcmp(t, KSPIBCGS) == 0)
    return Ksp::Ibcgs;
  else if (strcmp(t, KSPFBCGS) == 0)
    return Ksp::Fbcgs;
  else if (strcmp(t, KSPFBCGSR) == 0)
    return Ksp::Fbcgsr;
  else if (strcmp(t, KSPBCGSL) == 0)
    return Ksp::Bcgsl;
  else if (strcmp(t, KSPCGS) == 0)
    return Ksp::Cgs;
  else if (strcmp(t, KSPTFQMR) == 0)
    return Ksp::Tfqmr;
  else if (strcmp(t,KSPCR) == 0)
    return Ksp::Cr;
  else if (strcmp(t,KSPPIPECR) == 0)
    return Ksp::Pipecr;
  else if (strcmp(t,KSPLSQR) == 0)
    return Ksp::Lsqr;
  else if (strcmp(t, KSPPREONLY) == 0)
    return Ksp::Preonly;
  else if (strcmp(t, KSPQCG) == 0)
    return Ksp::Qcg;
  else if (strcmp(t, KSPBICG) == 0)
    return Ksp::Bicg;
  else if (strcmp(t, KSPMINRES) == 0)
    return Ksp::Minres;
  else if (strcmp(t, KSPSYMMLQ) == 0)
    return Ksp::Symmlq;
  else if (strcmp(t, KSPLCD) == 0)
    return Ksp::Lcd;
  else if (strcmp(t, KSPPYTHON) == 0)
    return Ksp::Python;
  else if (strcmp(t, KSPGCR) == 0)
    return Ksp::Gcr;
  else if (strcmp(t, KSPSPECEST) == 0)
    return Ksp::Specest;

  return Ksp::Invalid;
}



Vector
Ksp::
solve(Matrix const& A, Vector const& b){
  Petscpp::Vector x = Petscpp::duplicate(b);
  KSPSetOperators(ksp_, A.petscMat(), A.petscMat(), DIFFERENT_NONZERO_PATTERN);
  KSPSolve(ksp_, b.petscVec(), x.petscVec());
  return x;
}


void
Ksp::
loadFromOptions(){
  KSPSetFromOptions(ksp_);
}


PPreconditioner
Ksp::
preconditioner() const{
  PC pc;
  KSPGetPC(ksp_, &pc);

  // TODO: Need a different handler or ctor for use-cases such as this
  PPreconditioner precond(PETSC_COMM_SELF);
  precond.pc_ = pc;
  precond.destroy_ = false;
  return precond;
}


void
Ksp::
setExternalPackage(SolverPackage package){
  preconditioner().setType(PPreconditioner::Lu);
  KSPSetType(ksp_, KSPPREONLY);
  preconditioner().setExternalPackage(package);
}


void
Ksp::
setPreconditionerType(PPreconditioner::Type p){
  preconditioner().setType(p);
}


/* ----------------------------------------------------------------- */

PPreconditioner::
PPreconditioner(MPI_Comm com)
  : destroy_(true)
{
  PCCreate(com, &pc_);
}


PPreconditioner::
PPreconditioner(PC&& pc)
  : pc_(nullptr), destroy_(true)
{
  std::swap(pc, pc_);
}


PPreconditioner::
PPreconditioner(PC pc)
  : pc_(pc), destroy_(false)
{ }


PPreconditioner::
~PPreconditioner(){
  if (destroy_)
    PCDestroy(&pc_);
}


PPreconditioner::Type
PPreconditioner::
type() const{
  PCType t;
  PCGetType(pc_, &t);

  if (strcmp(t,PCNONE) == 0)
    return PPreconditioner::None;
  else if (strcmp(t, PCJACOBI) == 0)
    return PPreconditioner::Jacobi;
  else if (strcmp(t, PCSOR) == 0)
    return PPreconditioner::Sor;
  else if (strcmp(t, PCLU) == 0)
    return PPreconditioner::Lu;
  else if (strcmp(t, PCSHELL) == 0)
    return PPreconditioner::Shell;
  else if (strcmp(t, PCBJACOBI) == 0)
    return PPreconditioner::Bjacobi;
  else if (strcmp(t, PCMG) == 0)
    return PPreconditioner::Mg;
  else if (strcmp(t, PCEISENSTAT) == 0)
    return PPreconditioner::Eisenstat;
  else if (strcmp(t, PCILU) == 0)
    return PPreconditioner::Ilu;
  else if (strcmp(t, PCICC) == 0)
    return PPreconditioner::Icc;
  else if (strcmp(t, PCASM) == 0)
    return PPreconditioner::Asm;
  else if (strcmp(t, PCGASM) == 0)
    return PPreconditioner::Gasm;
  else if (strcmp(t, PCKSP) == 0)
    return PPreconditioner::Ksp;
  else if (strcmp(t, PCCOMPOSITE) == 0)
    return PPreconditioner::Composite;
  else if (strcmp(t, PCREDUNDANT) == 0)
    return PPreconditioner::Redundant;
  else if (strcmp(t, PCSPAI) == 0)
    return PPreconditioner::Spai;
  else if (strcmp(t, PCNN) == 0)
    return PPreconditioner::Nn;
  else if (strcmp(t, PCCHOLESKY) == 0)
    return PPreconditioner::Cholesky;
  else if (strcmp(t, PCPBJACOBI) == 0)
    return PPreconditioner::Pbjacobi;
  else if (strcmp(t, PCMAT) == 0)
    return PPreconditioner::Mat;
  else if (strcmp(t, PCHYPRE) == 0)
    return PPreconditioner::Hypre;
  else if (strcmp(t, PCPARMS) == 0)
    return PPreconditioner::Parms;
  else if (strcmp(t, PCFIELDSPLIT) == 0)
    return PPreconditioner::Fieldsplit;
  else if (strcmp(t, PCTFS) == 0)
    return PPreconditioner::Tfs;
  else if (strcmp(t, PCML) == 0)
    return PPreconditioner::Ml;
  else if (strcmp(t, PCGALERKIN) == 0)
    return PPreconditioner::Galerkin;
  else if (strcmp(t, PCEXOTIC) == 0)
    return PPreconditioner::Exotic;
  else if (strcmp(t, PCHMPI) == 0)
    return PPreconditioner::Hmpi;
  else if (strcmp(t, PCSUPPORTGRAPH) == 0)
    return PPreconditioner::Supportgraph;
  else if (strcmp(t, PCASA) == 0)
    return PPreconditioner::Asa;
  else if (strcmp(t, PCCP) == 0)
    return PPreconditioner::Cp;
  else if (strcmp(t, PCBFBT) == 0)
    return PPreconditioner::Bfbt;
  else if (strcmp(t, PCLSC) == 0)
    return PPreconditioner::Lsc;
  else if (strcmp(t, PCPYTHON) == 0)
    return PPreconditioner::Python;
  else if (strcmp(t, PCPFMG) == 0)
    return PPreconditioner::Pfmg;
  else if (strcmp(t, PCSYSPFMG) == 0)
    return PPreconditioner::Syspfmg;
  else if (strcmp(t, PCREDISTRIBUTE) == 0)
    return PPreconditioner::Redistribute;
  else if (strcmp(t, PCSVD) == 0)
    return PPreconditioner::Svd;
  else if (strcmp(t, PCGAMG) == 0)
    return PPreconditioner::Gamg;
  else if (strcmp(t, PCSACUSP) == 0)
    return PPreconditioner::Sacusp;
  else if (strcmp(t, PCSACUSPPOLY) == 0)
    return PPreconditioner::Sacusppoly;
  else if (strcmp(t, PCBICGSTABCUSP) == 0)
    return PPreconditioner::Bicgstabcusp;
  else if (strcmp(t, PCAINVCUSP) == 0)
    return PPreconditioner::Ainvcusp;
  else if (strcmp(t, PCBDDC) == 0)
    return PPreconditioner::Bddc;

  return PPreconditioner::None;
}


void
PPreconditioner::
setExternalPackage(SolverPackage package){
  std::string str = toString(package);
  PCFactorSetMatSolverPackage(pc_, str.c_str());
}


std::string
Petscpp::
toString(PPreconditioner::Type t){
  switch (t){
  case (PPreconditioner::None):
    return PCNONE;
  case (PPreconditioner::Jacobi):
    return PCJACOBI;
  case (PPreconditioner::Sor):
    return PCSOR;
  case (PPreconditioner::Lu):
    return PCLU;
  case (PPreconditioner::Shell):
    return PCSHELL;
  case (PPreconditioner::Bjacobi):
    return PCBJACOBI;
  case (PPreconditioner::Mg):
    return PCMG;
  case (PPreconditioner::Eisenstat):
    return PCEISENSTAT;
  case (PPreconditioner::Ilu):
    return PCILU;
  case (PPreconditioner::Icc):
    return PCICC;
  case (PPreconditioner::Asm):
    return PCASM;
  case (PPreconditioner::Gasm):
    return PCGASM;
  case (PPreconditioner::Ksp):
    return PCKSP;
  case (PPreconditioner::Composite):
    return PCCOMPOSITE;
  case (PPreconditioner::Redundant):
    return PCREDUNDANT;
  case (PPreconditioner::Spai):
    return PCSPAI;
  case (PPreconditioner::Nn):
    return PCNN;
  case (PPreconditioner::Cholesky):
    return PCCHOLESKY;
  case (PPreconditioner::Pbjacobi):
    return PCPBJACOBI;
  case (PPreconditioner::Mat):
    return PCMAT;
  case (PPreconditioner::Hypre):
    return PCHYPRE;
  case (PPreconditioner::Parms):
    return PCPARMS;
  case (PPreconditioner::Fieldsplit):
    return PCFIELDSPLIT;
  case (PPreconditioner::Tfs):
    return PCTFS;
  case (PPreconditioner::Ml):
    return PCML;
  case (PPreconditioner::Galerkin):
    return PCGALERKIN;
  case (PPreconditioner::Exotic):
    return PCEXOTIC;
  case (PPreconditioner::Hmpi):
    return PCHMPI;
  case (PPreconditioner::Supportgraph):
    return PCSUPPORTGRAPH;
  case (PPreconditioner::Asa):
    return PCASA;
  case (PPreconditioner::Bfbt):
    return PCBFBT;
  case (PPreconditioner::Lsc):
    return PCLSC;
  case (PPreconditioner::Python):
    return PCPYTHON;
  case (PPreconditioner::Pfmg):
    return PCPFMG;
  case (PPreconditioner::Syspfmg):
    return PCSYSPFMG;
  case (PPreconditioner::Redistribute):
    return PCREDISTRIBUTE;
  case (PPreconditioner::Svd):
    return PCSVD;
  case (PPreconditioner::Gamg):
    return PCGAMG;
  case (PPreconditioner::Sacusp):
    return PCSACUSP;
  case (PPreconditioner::Sacusppoly):
    return PCSACUSPPOLY;
  case (PPreconditioner::Bicgstabcusp):
    return PCBICGSTABCUSP;
  case (PPreconditioner::Ainvcusp):
    return PCAINVCUSP;
  case (PPreconditioner::Bddc):
    return PCBDDC;
  default:
    return "";
  }
}


void
PPreconditioner::
setType(PPreconditioner::Type type){
  std::string str = toString(type);
  if (str != "")
    PCSetType(pc_, str.c_str());
}


SolverPackage from_string(std::string str){
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == MATSOLVERSUPERLU)
    return SolverPackage::SuperLU;
  else if (str == MATSOLVERSUPERLU_DIST)
    return SolverPackage::SuperLU_dist;
  else if (str == MATSOLVERUMFPACK)
    return SolverPackage::Umfpack;
  else if (str == MATSOLVERCHOLMOD)
    return SolverPackage::Cholmod;
  else if (str == MATSOLVERESSL)
    return SolverPackage::Essl;
  else if (str == MATSOLVERLUSOL)
    return SolverPackage::Lusol;
  else if (str == MATSOLVERMUMPS)
    return SolverPackage::Mumps;
  else if (str == MATSOLVERPASTIX)
    return SolverPackage::Pastix;
  else if (str == MATSOLVERMATLAB)
    return SolverPackage::Matlab;
  else if (str == MATSOLVERPETSC)
    return SolverPackage::Petsc;
  else if (str == MATSOLVERBAS)
    return SolverPackage::Verbas;
  else if (str == MATSOLVERCUSPARSE)
    return SolverPackage::CuSparse;
  else if (str == MATSOLVERBSTRM)
    return SolverPackage::Bstrm;
  else if (str == MATSOLVERSBSTRM)
    return SolverPackage::Sbstrm;
  else if (str == MATSOLVERELEMENTAL)
    return SolverPackage::Elemental;
  else if (str == MATSOLVERCLIQUE)
    return SolverPackage::Clique;

  return SolverPackage::Invalid;
}


SolverPackage
PPreconditioner::
externalPackage() const{
  MatSolverPackage package;
  PCFactorGetMatSolverPackage(pc_, const_cast<const MatSolverPackage*>(&package));

  if (package == nullptr)
    return from_string("");
  else
    return from_string(package);
}


std::string
Petscpp::
toString(SolverPackage package){
  switch (package){
  case (SolverPackage::SuperLU):
    return MATSOLVERSUPERLU;
  case (SolverPackage::SuperLU_dist):
    return MATSOLVERSUPERLU_DIST;
  case (SolverPackage::Umfpack):
    return MATSOLVERUMFPACK;
  case (SolverPackage::Cholmod):
    return MATSOLVERCHOLMOD;
  case (SolverPackage::Essl):
    return MATSOLVERESSL;
  case (SolverPackage::Lusol):
    return MATSOLVERLUSOL;
  case (SolverPackage::Mumps):
    return MATSOLVERMUMPS;
  case (SolverPackage::Pastix):
    return MATSOLVERPASTIX;
  case (SolverPackage::Matlab):
    return MATSOLVERMATLAB;
  case (SolverPackage::Petsc):
    return MATSOLVERPETSC;
  case (SolverPackage::Verbas):
    return MATSOLVERBAS;
  case (SolverPackage::CuSparse):
    return MATSOLVERCUSPARSE;
  case (SolverPackage::Bstrm):
    return MATSOLVERBSTRM;
  case (SolverPackage::Sbstrm):
    return MATSOLVERSBSTRM;
  case (SolverPackage::Elemental):
    return MATSOLVERELEMENTAL;
  case (SolverPackage::Clique):
    return MATSOLVERCLIQUE;
  default:
    return "";
  }
}


void
Petscpp::
printInfo(Ksp const& ksp, std::ostream &stream){
  if (procId() == 0){
    std::string solver = toString(ksp.preconditioner().externalPackage());
    if (solver != ""){
      stream << "Solver package = " << solver << '\n';
    }
    else{
      stream << "Solver package = N/A\n";
    }

    std::string pc = toString(ksp.preconditioner().type());
    if (pc != "")
      stream << "Preconditioner type = " << pc << '\n';
    else
      stream << "Preconditioner type = N/A\n";

    std::string ksp_str = toString(ksp.type());
    if (ksp_str != "")
      stream << "KSP type = " << ksp_str << '\n';
    else
      stream << "KSP type = N/A\n";
  }
}
