#ifndef VECR2_H
#define VECR2_H
#pragma once
class VecR2{
private:
	double x, y;
public:
	VecR2(); //Default
	~VecR2(); //Destruct
	VecR2(const VecR2& vec);
	VecR2& operator=(const VecR2& rhs);
	VecR2(VecR2&& vec);
	VecR2& operator=(VecR2&& rhs);

	friend std::ostream &operator<<(std::ostream &os, VecR2 const &rhs);

	VecR2(double a, double b);

	VecR2 VecR2::operator+(const VecR2& rhs);
	VecR2 VecR2::operator-(const VecR2& rhs);
	double VecR2::operator*(const VecR2& rhs);
	VecR2 VecR2::operator*(double rhs);
	friend VecR2 operator*(double lhs, const VecR2& rhs);



	double getX() { return x; }
	double getY() { return y; }

	void setX(double X) { x = X; }
	void setY(double Y) { y = Y; }
};
#endif
