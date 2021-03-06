#ifndef HEATTRANSFER1D_H
#define HEATTRANSFER1D_H

#include <petsc.h>

#include <petsc/private/vecimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/matimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/dmimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/tsimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!

#define NODES_SIZE 1
#define PIPES_SIZE 4
#define DMS_SIZE PIPES_SIZE + NODES_SIZE

typedef struct Params {
    double* temperature_presc_;
    double  conductivity_;
    double  source_term_;
    double  wall_length_;
} Params;

typedef struct {
  PetscScalar T;
} Field;

PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p);
PetscErrorCode FormFunctionSinglePipe(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p);
PetscErrorCode FormCoupleLocations(DM dmcomposite, Mat A, PetscInt *dnz, PetscInt *onz, PetscInt __rstart, PetscInt __nrows, PetscInt __start, PetscInt __end);
PetscErrorCode CompositeSetCoupling(DM dm);
PetscErrorCode RedundantSetSize(DM dm, PetscMPIInt rank, PetscInt N);

#endif /* !HEATTRANSFER1D_H */
