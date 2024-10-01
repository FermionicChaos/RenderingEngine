#ifndef POINTR4_H
#define POINTR4_H

enum basis_t {
	FLAT,
	CYLINDRICAL,
	SPHERICAL,
	ELLIPTICAL,
	SCHWARZSCHILD,
	KERR
};

enum type_t {
	COVARIANT,
	CONTRAVARIANT
};

class PointR4 {
public:
	float x0, x1, x2, x3;
	type_t type;
	basis_t basis;

	PointR4();
	~PointR4();

	PointR4(float t, float x, float y, float z);


};

#endif // !POINTR4_H
