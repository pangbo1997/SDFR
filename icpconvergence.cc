/*
Szymon Rusinkiewicz
Princeton University

icprobustness.cc
Conduct an experiment to test ICP robustness under perturbations to initial
translation/rotation.
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include "TriMesh.h"
#include "TriMesh_algo.h"
#include "ICP_align.inc"
#include "ICP_iter.inc"
using namespace std;
using namespace trimesh;


// ICP Parameters
#define DESIRED_PAIRS 200
#define APPROX_EPS 0.05f
#define REJECTION_THRESH_MULT 3.7f // convert median to sigma, then times 2.5
#define LAMBDA_INITIAL 1.0f
#define LAMBDA_MIN 1.0e-6f
#define LAMBDA_MAX 1.0e3f
#define LM_FACTOR 2.0f


// Experiment Parameters
#define MAX_ANGLE 60
#define ANGLE_STEPS 4
#define MAX_TRANS 0.5f
#define TRANS_STEPS 6
#define NMAXITERS 3
const int maxiters[NMAXITERS] = { 20, 100, 500 };
#define SUCCESS_THRESH 0.01f


// Compute average ground-truth vertex error produced by this xform.
// Uses min(DESIRED_PAIRS, #verts) vertices.
static float err(const TriMesh *mesh, const xform &xf)
{
	float sum = 0.0f;
	size_t nv = mesh->vertices.size();
	size_t incr = max((size_t) 1, nv / DESIRED_PAIRS);
	for (size_t i = 0; i < nv; i += incr)
		sum += dist2(mesh->vertices[i], xf * mesh->vertices[i]);
	return sqrt(sum / (nv / incr));
}


// Trivial implementation of ICP that's really only intended for tests
static int ICP_until_converged(const TriMesh *mesh1, const TriMesh *mesh2,
	                       const KDtree *kd1, const KDtree *kd2,
			       xform &xf2, float maxdist,
			       ICP_Variant variant)
{
	float size = mesh1->bsphere.r + mesh2->bsphere.r;
	float cdfincr = 2.0f / DESIRED_PAIRS;

	float best_err = 3.3e33f;
	int iter_of_best_err = 0;

	// For VARIANT_LM
	float lambda = LAMBDA_INITIAL;

	int iter = 0;
	while (iter++ < 300) {

		bool success = ICP_iter(mesh1, mesh2, xform(), xf2, kd1, kd2,
			cdfincr, DESIRED_PAIRS, APPROX_EPS, maxdist,
			0.0f, REJECTION_THRESH_MULT, variant, lambda);

		// Check if converged
		float this_err = err(mesh2, xf2);
		if (this_err < SUCCESS_THRESH * size)
			return iter;
		if (this_err < best_err) {
			best_err = this_err;
			iter_of_best_err = iter;
		}
		if (iter - iter_of_best_err > 50)
			return maxiters[NMAXITERS-1];

		if (variant == VARIANT_PT2PL_ONEWAY_LM ||
		    variant == VARIANT_SYMM_LM) {
			if (lambda > LAMBDA_MIN)
				lambda /= LM_FACTOR;
		}
	}

	return iter;
}


// Converts pairs of random numbers in [0,1] to
// uniformly-distributed points on the sphere
template <class T = float>
static inline Vec<3,T> rnd2sph(const T &u, const T &v)
{
	T phi = u * T(M_2PI);
	T z = v * 2 - 1;
	T xy = sqrt(T(1) - sqr(z));
	return Vec<3,T>(xy * cos(phi), xy * sin(phi), z);
}


void usage(const char *myname)
{
	fprintf(stderr, "Usage: %s in1.ply in2.ply [variant] [ntrials]\n", myname);
	exit(1);
}

int main(int argc, char *argv[])
{
	if (argc < 3)
		usage(argv[0]);

	TriMesh *mesh1 = TriMesh::read(argv[1]);
	if (!mesh1)
		usage(argv[0]);

	TriMesh *mesh2 = TriMesh::read(argv[2]);
	if (!mesh2)
		usage(argv[0]);

	ICP_Variant variant = VARIANT_SYMM;
	if (argc > 3)
		variant = (ICP_Variant) atoi(argv[3]);
	if (variant < 0 || variant > VARIANT_SYMM_LM)
		usage(argv[0]);

	int ntrials = 10;
	if (argc > 4)
		ntrials = max(1, atoi(argv[4]));

	// Apply xforms
	xform xf1, xf2;
	xf1.read(xfname(argv[1]));
	apply_xform(mesh1, xf1);

	xf2.read(xfname(argv[2]));
	apply_xform(mesh2, xf2);

	// Compute mesh properties
	mesh1->need_normals();
	mesh2->need_normals();
	if (variant == VARIANT_MITRA)
		mesh1->need_curvatures();

	KDtree *kd1 = new KDtree(mesh1->vertices);
	KDtree *kd2 = new KDtree(mesh2->vertices);

	// Compute rotcenter, size
	mesh1->need_bsphere();
	mesh2->need_bsphere();
	point rotcenter = 0.5f * (mesh1->bsphere.center + mesh2->bsphere.center);
	float size = mesh1->bsphere.r + mesh2->bsphere.r;

	float max_angle = radians(MAX_ANGLE);
	float max_trans = MAX_TRANS * size;
	float initial_maxdist = size + max_trans;

	// Start the trials
	int nsuccess[ANGLE_STEPS][TRANS_STEPS][NMAXITERS];

	char filename[256];
	memset(nsuccess, 0, sizeof(nsuccess));
	for (int trial = 0; trial < ntrials; trial++) {
		printf("%d:\n", trial + 1);
		// All of the initial transforms for a single trial will have
		// rotations about the same axis, and multiples of same trans



        xform thisxf2;
        // Do ICP
        int iters = ICP_until_converged(mesh1, mesh2, kd1, kd2,
            thisxf2, initial_maxdist, variant);
        // Record whether it succeeded within # of iters

		snprintf(filename,sizeof(filename),"%s_trans.txt",argv[5]);
        thisxf2.write(filename);


	}


}

