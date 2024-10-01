#include "PointR4.h"

PointR4::PointR4(){
	 
}

PointR4::~PointR4() {

}

PointR4::PointR4(float t, float x, float y, float z) {
	type = CONTRAVARIANT;
	basis = FLAT;
	x0 = t; x1 = x, x2 = y; x3 = z;
}
