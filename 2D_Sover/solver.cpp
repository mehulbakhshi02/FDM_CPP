#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <functional>
#include <petscksp.h> //mpicxx mycode.cpp -I/usr/include/petsc -lpetsc

using namespace std;

class  HeatSolver {
private:
	double L;
	double a; // thermal diffusivity
	int N;
	double dt;
	int steps;

	vector<double> T;
	function<double(double,double)> source;

public:
	HeatSolver(double length, double alpha, int nodes, double timestep, double timesteps)
	: L(length), a(alpha), N(nodes), dt(timestep), steps(timesteps)
	{
		T.resize(N);
	}

	// Boundary conditions
	void setBoundary(double left, double right) {
		T[0] = left;
		T[N-1] = right;
	}

	// Initial condition
	void setInitial(double temp) {
		for (int i=1; i<N-1; i++) {
			T[i] = temp;
		}
	}

	// Source term
	void setSource(function<double(double,double)> q) {
		source = q;
	}

	void solve_exp() {
			double dx = L/(N-1);
			double r = a * dt / (dx * dx);

			vector<double> T_new(N);
			for (int n=0; n<steps; n++) {

				double t = n * dt;

				// Apply boundary conditions
				T_new[0] = T[0];
				T_new[N-1] = T[N-1];

				// Update interior points
				for (int i=1; i<N-1; i++) {
					double xpos = i * dx;
					double q_inst;
					if (source) {
						q_inst = source(xpos,t);
					}
					else {
						q_inst = 0.0;
					}
					
					T_new[i] = T[i] + r * (T[i+1] - 2*T[i] + T[i-1]) + dt * q_inst;
				}

				// update T
				T = T_new;
			}
		}

	void solve_imp()
	{
		double dx = L / (N - 1);
		double r  = a * dt / (dx * dx);

		// ---- PETSc initialization ----
		Mat A;
		Vec x, b;
		KSP ksp;
		PetscInt n = N;

		MatCreate(PETSC_COMM_WORLD, &A);
		MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n);
		MatSetFromOptions(A);
		MatSetUp(A);

		// Build the coefficient matrix A (time-independent)
		for (int i = 1; i < N - 1; i++) {
			MatSetValue(A, i, i-1, -r, INSERT_VALUES);
			MatSetValue(A, i, i, 1.0 + 2.0*r, INSERT_VALUES);
			MatSetValue(A, i, i+1, -r, INSERT_VALUES);
		}

		// Boundary conditions
		MatSetValue(A, 0, 0, 1.0, INSERT_VALUES);
		MatSetValue(A, N-1, N-1, 1.0, INSERT_VALUES);

		MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

		// RHS(b) and solution(x)
		VecCreate(PETSC_COMM_WORLD, &x);
		VecSetSizes(x, PETSC_DECIDE, n);
		VecSetFromOptions(x);

		VecDuplicate(x, &b);

		// Linear solver
		KSPCreate(PETSC_COMM_WORLD, &ksp);
		KSPSetOperators(ksp, A, A);
		KSPSetFromOptions(ksp);   // allows -ksp_type gmres or cg etc. from CLI

		// Time stepping loop
		for (int n = 0; n < steps; n++) {

			double t = n * dt;

			// Fill RHS vector b with old T values
			for (int i = 0; i < N; i++){
				double xpos = i * dx;
				double q_inst;
				if (source) {
					q_inst = source(xpos,t);
				}
				else {
					q_inst = 0.0;
				}

				VecSetValue(b, i, T[i]+dt*q_inst, INSERT_VALUES);
			}

			// Apply boundary conditions to b
			VecSetValue(b, 0, T[0], INSERT_VALUES);
			VecSetValue(b, N-1, T[N-1], INSERT_VALUES);

			VecAssemblyBegin(b);
			VecAssemblyEnd(b);

			// Solve A x = b
			KSPSolve(ksp, b, x);

			// Copy solution back into T
			for (int i = 0; i < N; i++) {
				double val;
				VecGetValues(x, 1, (const PetscInt*)&i, &val);
				T[i] = val;
			}
		}

		// Cleanup
		KSPDestroy(&ksp);
		VecDestroy(&x);
		VecDestroy(&b);
		MatDestroy(&A);
	}
	
	const vector<double> getSolution() {
		return T; // Return the temperature distribution at final time step
	}
};