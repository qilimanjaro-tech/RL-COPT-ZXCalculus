OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
rz(-3*pi/2) q[0];
rz(-3*pi/2) q[1];
rz(-3*pi/2) q[2];
rz(-3*pi/2) q[3];
ry(pi/2) q[4];
rx(pi) q[4];
rzz(pi/2) q[3],q[4];
ry(pi/2) q[3];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[3];
rz(-pi/2) q[3];
ry(-pi/2) q[3];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rzz(pi/2) q[3],q[4];
rz(-pi/2) q[3];
ry(pi/2) q[3];
rzz(pi/2) q[2],q[3];
ry(pi/2) q[2];
rz(-pi/2) q[3];
ry(-pi/2) q[3];
rzz(pi/2) q[3],q[2];
rz(-pi/2) q[2];
ry(-pi/2) q[2];
rz(-pi/2) q[3];
ry(pi/2) q[3];
rzz(pi/2) q[2],q[3];
rz(-pi/2) q[2];
ry(pi/2) q[2];
rzz(pi/2) q[1],q[2];
ry(pi/2) q[1];
rz(-pi/2) q[2];
ry(-pi/2) q[2];
rzz(pi/2) q[2],q[1];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rz(-pi/2) q[2];
ry(pi/2) q[2];
rzz(pi/2) q[1],q[2];
rz(-pi) q[1];
rz(pi/2) q[2];
rz(-pi/2) q[3];
ry(-pi/2) q[3];
rz(-pi) q[4];
rzz(pi/2) q[0],q[4];
ry(pi/2) q[0];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[0];
rz(-pi/2) q[0];
ry(-pi/2) q[0];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
rz(pi/2) q[0];
ry(-pi/2) q[0];
rz(pi) q[4];
ry(-pi/2) q[5];
rz(-pi) q[5];
rzz(pi/2) q[3],q[5];
rz(-pi/2) q[3];
rzz(pi/2) q[2],q[3];
rzz(pi/2) q[1],q[3];
rzz(pi/2) q[0],q[3];
rz(-pi) q[0];
ry(pi/2) q[1];
ry(pi/2) q[3];
rzz(pi/2) q[2],q[3];
ry(-pi/2) q[2];
rzz(pi/2) q[2],q[5];
rz(-1.5835739592902778) q[2];
rz(1.5580186942995153) q[3];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[1];
rz(0.01277763249538122) q[0];
rzz(pi/2) q[4],q[1];
rz(-4.725166612880072) q[5];
rzz(pi/2) q[0],q[5];
ry(-pi/2) q[0];
rzz(pi/2) q[3],q[5];
rzz(pi/2) q[2],q[5];
rzz(pi/2) q[0],q[5];
rzz(pi/2) q[0],q[2];
ry(-pi/2) q[0];
ry(pi/2) q[2];
rzz(pi/2) q[3],q[4];
ry(-pi/2) q[3];
ry(pi/2) q[4];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
rz(-3.1288150210943915) q[0];
rzz(pi/2) q[0],q[4];
rzz(pi/2) q[0],q[2];
rx(3.135130873586045) q[0];
rz(pi) q[2];
rz(pi/2) q[4];
ry(pi/2) q[4];
rz(pi/2) q[5];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
ry(1.577258106798645) q[0];
rz(-pi) q[0];
rzz(pi/2) q[3],q[5];
rz(-pi/2) q[3];
ry(pi/2) q[3];
rzz(pi/2) q[2],q[3];
ry(-pi/2) q[2];
rz(0.01277763249538122) q[3];
rzz(pi/2) q[0],q[3];
ry(pi/2) q[3];
ry(pi/2) q[5];
rzz(pi/2) q[2],q[5];
rz(-0.01277763249538122) q[2];
rzz(pi/2) q[2],q[3];
ry(-pi/2) q[2];
rz(pi/2) q[3];
ry(pi/2) q[3];
rz(pi/2) q[5];
ry(pi/2) q[5];
rzz(pi/2) q[2],q[5];
rz(-pi/2) q[2];
ry(pi/2) q[2];
rz(0.012777632495380331) q[5];
rzz(pi/2) q[0],q[5];
ry(-3.1351308735860455) q[0];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[1];
rz(0.0064617800037480855) q[5];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[2];
rz(-3*pi/2) q[2];
ry(0.006461780003748332) q[5];
rz(-pi/2) q[5];
rzz(pi/2) q[5],q[3];
rz(-pi) q[3];
ry(3.1351308735860446) q[5];
rz(-pi) q[5];
rzz(pi/2) q[5],q[2];
rzz(pi/2) q[0],q[2];
ry(-pi/2) q[0];
ry(-pi/2) q[2];
ry(pi/2) q[5];
rzz(pi/2) q[2],q[5];
rz(-pi/2) q[2];
ry(pi/2) q[2];
rz(3.1351308735860446) q[5];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[4];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[3];
ry(-pi/2) q[3];
rzz(pi/2) q[3],q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[2];
rz(-pi/2) q[1];
ry(pi/2) q[1];
rz(pi/2) q[2];
ry(pi/2) q[2];
rzz(pi/2) q[0],q[2];
rz(1.577258106798645) q[0];
ry(-pi/2) q[0];
rz(-pi) q[2];
rz(-0.012561759996077448) q[3];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[1];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rz(-0.012561759996078337) q[4];
rzz(pi/2) q[1],q[4];
rz(3*pi/2) q[1];
rzz(pi/2) q[0],q[1];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[2];
rz(-3*pi/2) q[0];
rzz(pi/2) q[1],q[5];
rzz(pi/2) q[0],q[1];
ry(pi/2) q[1];
ry(pi/2) q[2];
ry(pi/2) q[4];
rzz(pi/2) q[3],q[4];
ry(-pi/2) q[3];
rz(-pi) q[4];
rzz(pi/2) q[3],q[4];
rz(-pi) q[3];
ry(pi/2) q[4];
ry(pi/2) q[5];
