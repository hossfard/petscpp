#ifndef PKSP_H_
#define PKSP_H_

#include "Vector.h"
#include "Matrix.h"
#include <string>
#include <iostream>

struct _p_KSP;
typedef struct _p_KSP* KSP;

struct _p_PC;
typedef struct _p_PC* PC;

namespace Petscpp{

  enum class SolverPackage{
    SuperLU, SuperLU_dist, Umfpack, Cholmod, Essl, Lusol,
    Mumps, Pastix, Matlab, Petsc, Verbas, CuSparse, Bstrm,
    Sbstrm, Elemental, Clique, Invalid
  };

  std::string toString(SolverPackage package);

  class PPreconditioner
  {
  public:
    enum Type{
      None, Jacobi, Sor, Lu, Shell, Bjacobi, Mg, Eisenstat, Ilu,
      Icc, Asm, Gasm, Ksp, Composite, Redundant, Spai, Nn, Cholesky,
      Pbjacobi, Mat, Hypre, Parms, Fieldsplit, Tfs, Ml, Galerkin, Exotic,
      Hmpi, Supportgraph, Asa, Cp, Bfbt, Lsc, Python, Pfmg, Syspfmg, Redistribute,
      Svd, Gamg, Sacusp, Sacusppoly, Bicgstabcusp, Ainvcusp, Bddc};

    PPreconditioner(MPI_Comm = PETSC_COMM_WORLD);
    PPreconditioner(PC&& pc);
    PPreconditioner(PC pc);
    ~PPreconditioner();

    Type type() const;
    void setExternalPackage(SolverPackage package);
    SolverPackage externalPackage() const;
    void setType(Type type);

  private:
    friend class Ksp;
    PC pc_;
    bool destroy_;
  };


  class Ksp
  {
  public:
    enum Type{
      Richardson, Chebyshev, Cg, Groppcg, Pipecg, Cgne, Nash, Stcg,
      Gltr, Gmres, Fgmres, Lgmres, Dgmres, Pgmres, Tcqmr, Bcgs,
      Ibcgs, Fbcgs, Fbcgsr, Bcgsl, Cgs, Tfqmr, Cr, Pipecr, Lsqr,
      Preonly, Qcg, Bicg, Minres, Symmlq, Lcd, Python, Gcr, Specest, Invalid
    };

    Ksp(Type type = Preonly, PPreconditioner::Type pcType = PPreconditioner::None);
    Ksp(Ksp const& other) = delete;
    Ksp& operator=(Ksp const& other) = delete;
    ~Ksp();

    /*! Return preconditioner type */
    PPreconditioner preconditioner() const;

    /*! Set the preconditioner for the KSP context */
    void setPreconditionerType(PPreconditioner::Type type);

    /*! Set KSP type */
    void setType(Type t);

    Type type() const;

    /*! Load settings specifed in configuration file
     *
     * Settings loaded from file overwrite any settings previously set
     * using the API
     */
    void loadFromOptions();

    /*! Specify direct solver package
     *
     * Internal preconditioner and KSP are set to 'PCLU' and
     * 'KSPPREONLY', respectively
     */
    void setExternalPackage(SolverPackage package);

    /*! Solve $A x = b$ */
    Vector solve(Matrix const& A, Vector const& b);

    KSP const& petscKsp() const;
    KSP& petscKsp();

  private:
    KSP ksp_;
  };

  /* Output preconditioner type, ksp type, and external pacakge info
     to stream */
  void printInfo(Ksp const& ksp, std::ostream &stream = std::cout);

  std::string toString(PPreconditioner::Type t);
  std::string toString(Ksp::Type t);

} /* namespace Petscpp */


#endif /* PKSP_H_ */
