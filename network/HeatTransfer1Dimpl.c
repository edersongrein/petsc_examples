#include "HeatTransfer1Dimpl.h"
#include "include_patch_pack.h"

#undef __FUNCT__
#define __FUNCT__ "RedundantSetSize"
PetscErrorCode RedundantSetSize(DM dm, PetscMPIInt rank, PetscInt N) {
    PetscErrorCode ierr;
    ierr = DMRedundantSetSize(dm, rank, N); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DM dm, PetscReal t, Field *x, Field *x_t, Field *f, Params *p, PetscInt comp_offset, PetscInt total_size)
 {
    PetscErrorCode ierr;
    PetscInt       xints,xinte,i;
    PetscReal      dx, L, k, conductivity, Q;
	DMDALocalInfo info;

    PetscFunctionBegin;

	ierr = DMDAGetLocalInfo(dm, &info); CHKERRQ(ierr);

	conductivity = p->conductivity_;
	Q = p->source_term_;
	L = p->wall_length_;

	/*
	Define mesh intervals ratios for uniform grid.
	*/
	dx = L / (PetscReal)(info.mx);
	k = conductivity;

	xints = info.xs;
	xinte = info.xs + info.xm;

	/* Compute over the interior points */
	for (i = xints; i<xinte; i++) {
		if (i == 0) {
			f[i].T = x_t[i].T * dx - 1.0 * (+k * (x[i + 1].T - x[i].T) / dx + Q * dx);
		}
		else if(i == info.mx) {
			f[i].T = x_t[i].T * dx - 1.0 * (-k * (x[i].T - x[i - 1].T) / dx + Q * dx);
		}
		else {
			f[i].T = x_t[i].T * dx
				- 1.0 * (+k * (x[i + 1].T - x[i].T) / dx
					- k * (x[i].T - x[i - 1].T) / dx + Q * dx);
		}
	}

	PetscFunctionReturn(0);
} 


#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p)
{
    DMDALocalInfo  info;
    PetscErrorCode ierr;
    DM             global_domain, dm;
    PetscInt nDM;

    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[DMS_SIZE];
    Vec Xs[DMS_SIZE], X_ts[DMS_SIZE], Fs[DMS_SIZE];
    Field* u_t[DMS_SIZE];
    Field* u[DMS_SIZE];
    Field* f[DMS_SIZE];

    PetscFunctionBegin;

    ierr = TSGetDM(ts, &global_domain); CHKERRQ(ierr);
	ierr = DMShellGetContext(global_domain, (void**)&dm);
	ierr = DMCompositeGetEntries(global_domain, &dm);
    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);

    /* Access the subvectors in X */    
    ierr = DMCompositeGetLocalVectorsArray(dm, nDM, NULL, Xs); CHKERRQ(ierr);
    ierr = DMCompositeScatterArray(dm, X, Xs); CHKERRQ(ierr);
    
    ierr = DMCompositeGetLocalVectorsArray(dm, nDM, NULL, X_ts); CHKERRQ(ierr);
    ierr = DMCompositeScatterArray(dm, X_t, X_ts); CHKERRQ(ierr);

    // Calculate total size
    PetscInt total_size = 0;
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
        total_size += info.mx;
    }
	total_size += 1;

    /* Access the subvectors in F.
    These are not ghosted so directly access the memory locations in F */
    ierr = DMCompositeGetAccessArray(dm, F, nDM, NULL, Fs); CHKERRQ(ierr);

    PetscInt comp_offset = 0;
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAVecGetArrayRead(das[i], X_ts[i], &(u_t[i])); CHKERRQ(ierr);
        ierr = DMDAVecGetArrayRead(das[i], Xs[i], &(u[i])); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(das[i], Fs[i], &(f[i])); CHKERRQ(ierr);
        
        // Central nodes
        ierr = FormFunctionLocal(das[i], t, u[i], u_t[i], f[i], p, comp_offset, total_size); CHKERRQ(ierr);
        comp_offset += info.mx;

		ierr = DMDAVecRestoreArrayRead(das[i], X_ts[i], &(u_t[i])); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArrayRead(das[i], Xs[i], &(u[i])); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(das[i], Fs[i], &(f[i])); CHKERRQ(ierr);
    }

	// Nodal part of residual
    ierr = VecGetArray(X_ts[PIPES_SIZE], &(u_t[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecGetArray(Xs[PIPES_SIZE], &(u[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecGetArray(Fs[PIPES_SIZE], &(f[PIPES_SIZE])); CHKERRQ(ierr);

    // Code goes here
	{
		PetscReal dx, L, k, conductivity,  Q;

		conductivity = p->conductivity_;
		Q = p->source_term_;
		L = p->wall_length_;
		k = conductivity;

		// Internal node
		f[PIPES_SIZE][0].T = 0.0; 
		double dx_avg = 0.0;

		for (int i = 0; i < PIPES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			int pipe_end = info.mx - 1;

			dx = L / (PetscReal)(info.mx);

			double flux_term = k * (u[PIPES_SIZE][0].T - u[i][pipe_end].T) / dx;
			f[PIPES_SIZE][0].T += +flux_term;
			f[i][pipe_end].T   += -flux_term;

			dx_avg += dx;
		}

		dx_avg /= PIPES_SIZE;

		double acc_term = u_t[PIPES_SIZE][0].T * dx_avg;
		double source_term = Q * dx_avg;
		f[PIPES_SIZE][0].T += acc_term - source_term;

		// bc's
		// left flux term w/ prescribed pressure
		for (int i = 0; i < PIPES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			dx = L / (PetscReal)(info.mx);

			double Tpresc = p->temperature_presc_[i];			

			int pipe_begin = 0;
			f[i][pipe_begin].T += -1.0 * (-k * (u[i][pipe_begin].T - Tpresc) / (0.5*dx));
		}
	}
    // end
    ierr = VecRestoreArray(X_ts[PIPES_SIZE], &(u_t[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecRestoreArray(Xs[PIPES_SIZE], &(u[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecRestoreArray(Fs[PIPES_SIZE], &(f[PIPES_SIZE])); CHKERRQ(ierr);

    ierr = DMCompositeRestoreAccessArray(dm, F, nDM, NULL, Fs); CHKERRQ(ierr);
    ierr = DMCompositeRestoreLocalVectorsArray(dm, nDM, NULL, Xs); CHKERRQ(ierr);
    ierr = DMCompositeRestoreLocalVectorsArray(dm, nDM, NULL, X_ts); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "CompositeSetCoupling"
PetscErrorCode CompositeSetCoupling(DM dm) {
    PetscErrorCode ierr = DMCompositeSetCoupling(dm, FormCoupleLocations);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "FormCoupleLocations"
/*
Computes the coupling between DA1 and DA2. This determines the location of each coupling between DA1 and DA2.
*/
PetscErrorCode FormCoupleLocations(DM dm, Mat A, PetscInt *dnz, PetscInt *onz, PetscInt __rstart, PetscInt __nrows, PetscInt __start, PetscInt __end)
{
    PetscInt       cols[3], row;
    PetscErrorCode ierr;
    PetscInt nDM;
    DMDALocalInfo info;
    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[DMS_SIZE];

    PetscFunctionBegin;
   
    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);

    // Calculate total size
    int size = 0;
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
        size += info.mx;
    }

	for (int i = nDM - NODES_SIZE; i < nDM; ++i) {
		PetscInt N;
		ierr = DMRedundantGetSize(das[i], NULL, &N); CHKERRQ(ierr);
		size += N;
	}
	
    // Hack: Bug in petsc file -> packm.c @ line (173) and line (129)
    // First A is NULL, then later dnz and onz are NULL, that's why
    // we need an IF here.
    if (!A) {
		int next_pipe_start_idx = 0;
		for (int i = 0; i < nDM - NODES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			next_pipe_start_idx += info.mx;

			cols[0] = next_pipe_start_idx - 1;
			row = size - 1;
			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);

			cols[0] = size - 1;
			row = next_pipe_start_idx - 1;
			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
		}
    }
    else {
        PetscScalar values[1];
		int next_pipe_start_idx = 0;

		for (int i = 0; i < nDM - NODES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			next_pipe_start_idx += info.mx;

			cols[0] = next_pipe_start_idx - 1;
			row = size - 1;
			values[0] = 0.0;
			ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);

			cols[0] = size - 1;
			row = next_pipe_start_idx - 1;
			values[0] = 0.0;
			ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
		}
    }

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionSinglePipe"
PetscErrorCode FormFunctionSinglePipe(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p)
{
	DMDALocalInfo  info;
	PetscErrorCode ierr;
	DM             dm;

	Vec X_local;
	Field* u_t;
	Field* u;
	Field* f;
	PetscInt local_size;

	PetscFunctionBegin;


	VecGetLocalSize(F, &local_size);

	ierr = TSGetDM(ts, &dm); CHKERRQ(ierr);

	//if (local_size > 0) {

		DMGetLocalVector(dm, &X_local);
		DMGlobalToLocalBegin(dm, X, INSERT_VALUES, X_local);
		DMGlobalToLocalEnd(dm, X, INSERT_VALUES, X_local);

		DMDAVecGetArrayRead(dm, X_local, &u);
		DMDAVecGetArray(dm, F, &f);
		DMDAVecGetArrayRead(dm, X_t, &u_t);

		FormFunctionLocal(dm, t, u, u_t, f, p, 0, 0);
		ierr = DMDAGetLocalInfo(dm, &info); CHKERRQ(ierr);

		PetscInt xints = info.xs;
		PetscInt xinte = info.xs + info.xm;

		double Tpresc_in = p->temperature_presc_[0];
		double Tpresc_out = p->temperature_presc_[1];

		if (xints == 0) {
			f[xints].T = Tpresc_in - u[xints].T;
		}
		if (xinte == info.mx) {
			f[xinte - 1].T = Tpresc_out - u[xinte - 1].T;
		}

		ierr = DMDAVecRestoreArrayRead(dm, X_t, &u_t); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArrayRead(dm, X_local, &u); CHKERRQ(ierr);
		ierr = DMDAVecRestoreArray(dm, F, &f); CHKERRQ(ierr);
		ierr = DMRestoreLocalVector(dm, &X_local); CHKERRQ(ierr);
	//}

	VecAssemblyBegin(F);
	VecAssemblyEnd(F);

	PetscFunctionReturn(0);
}
