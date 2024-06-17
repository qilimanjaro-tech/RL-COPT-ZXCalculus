OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
rz(-3*pi/2) q[0];
rz(-3*pi/2) q[1];
rz(-3*pi/2) q[4];
rz(-3*pi/2) q[5];
rz(-3*pi/2) q[6];
rz(-3*pi/2) q[8];
ry(pi/2) q[9];
rx(pi) q[9];
rzz(pi/2) q[8],q[9];
ry(pi/2) q[8];
rz(-pi/2) q[9];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[8];
rz(-pi/2) q[8];
ry(-pi/2) q[8];
rz(-pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[8],q[9];
rz(-pi/2) q[8];
ry(pi/2) q[8];
rz(-pi) q[9];
rzz(pi/2) q[6],q[9];
ry(pi/2) q[6];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[6];
rz(-pi/2) q[6];
ry(-pi/2) q[6];
rz(-pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[6],q[9];
rz(-pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
ry(pi/2) q[5];
rz(-pi/2) q[6];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[5];
rz(-pi/2) q[5];
ry(-pi/2) q[5];
rz(-pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rz(-pi/2) q[5];
ry(pi/2) q[5];
rzz(pi/2) q[1],q[5];
ry(pi/2) q[1];
rz(-pi/2) q[5];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[1];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rz(-pi/2) q[5];
ry(pi/2) q[5];
rzz(pi/2) q[1],q[5];
rz(pi/2) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[8];
rz(-7*pi/2) q[1];
rz(pi) q[5];
rz(-pi) q[6];
rzz(pi/2) q[4],q[6];
ry(pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[4];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
rz(-pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[4],q[6];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
ry(pi/2) q[0];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[0];
rz(-pi/2) q[0];
ry(-pi/2) q[0];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
rz(pi/2) q[0];
ry(-pi/2) q[0];
rz(pi/2) q[4];
rz(-pi/2) q[6];
rz(pi/2) q[8];
ry(pi/2) q[8];
rzz(pi/2) q[4],q[8];
rzz(pi/2) q[1],q[4];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[9];
rz(pi/2) q[4];
rz(-3*pi/2) q[8];
rzz(pi/2) q[1],q[8];
rzz(pi/2) q[1],q[7];
rzz(pi/2) q[1],q[6];
rzz(pi/2) q[6],q[8];
rz(pi) q[7];
ry(pi/2) q[9];
rzz(pi/2) q[5],q[9];
ry(pi/2) q[5];
rzz(pi/2) q[1],q[5];
rzz(pi/2) q[1],q[4];
rzz(pi/2) q[1],q[3];
rzz(pi/2) q[1],q[2];
rzz(pi/2) q[0],q[1];
rz(-pi/2) q[0];
rzz(pi/2) q[0],q[8];
ry(-pi/2) q[1];
rz(-3*pi/2) q[2];
rz(-pi/2) q[3];
rzz(pi/2) q[4],q[6];
rzz(pi/2) q[1],q[6];
rz(4.735340392883387) q[1];
ry(pi/2) q[4];
rz(3*pi/2) q[5];
ry(pi/2) q[6];
rzz(pi/2) q[0],q[6];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[4];
rz(-7.87693304647318) q[0];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
rz(1.5478449142961983) q[6];
rzz(pi/2) q[0],q[6];
ry(pi/2) q[6];
rzz(pi/2) q[4],q[6];
rz(4.689437567885991) q[4];
rzz(pi/2) q[0],q[4];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
rzz(pi/2) q[0],q[1];
ry(-pi/2) q[0];
rzz(pi/2) q[1],q[6];
rz(-pi) q[4];
rzz(pi/2) q[1],q[4];
rzz(pi/2) q[0],q[1];
ry(-pi/2) q[1];
ry(-pi/2) q[4];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rzz(pi/2) q[4],q[5];
rz(-3.1186412410910958) q[4];
ry(pi/2) q[5];
rz(-6.260233894680889) q[6];
rzz(pi/2) q[3],q[6];
rzz(pi/2) q[2],q[6];
rzz(pi/2) q[1],q[6];
rzz(pi/2) q[0],q[6];
ry(-pi/2) q[0];
rzz(pi/2) q[3],q[4];
rzz(pi/2) q[2],q[4];
rzz(pi/2) q[1],q[4];
ry(-pi/2) q[1];
ry(pi/2) q[3];
rzz(pi/2) q[2],q[3];
rz(-pi/2) q[3];
ry(-pi/2) q[3];
ry(pi/2) q[4];
ry(-pi/2) q[6];
rz(pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[6],q[9];
rz(-0.017777184998498896) q[6];
rzz(pi/2) q[1],q[6];
rz(1.593747739293594) q[1];
rzz(pi/2) q[1],q[4];
rz(pi/2) q[4];
ry(pi/2) q[4];
ry(pi/2) q[6];
rz(pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[0],q[9];
rz(-1.5937477392935957) q[0];
rzz(pi/2) q[0],q[8];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[1];
rz(-pi/2) q[0];
ry(pi/2) q[0];
ry(-pi/2) q[1];
ry(pi/2) q[8];
rz(pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[1],q[9];
rz(3.123815468591294) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[4];
rz(pi/2) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[6];
rz(1.5530191417963977) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[0];
rz(-pi/2) q[0];
ry(-pi/2) q[0];
rz(-pi/2) q[1];
ry(pi/2) q[1];
rz(-pi/2) q[4];
rzz(pi/2) q[0],q[4];
rz(-pi/2) q[0];
ry(pi/2) q[0];
rz(-pi/2) q[6];
ry(pi) q[6];
rzz(pi/2) q[6],q[1];
rz(1.5885735117933955) q[1];
ry(3.123815468591294) q[6];
rz(-pi/2) q[6];
rzz(pi/2) q[6],q[4];
ry(pi/2) q[4];
ry(3.123815468591294) q[6];
rz(-pi/2) q[6];
rz(-pi/2) q[9];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[5];
rz(-pi/2) q[5];
rzz(pi/2) q[9],q[8];
rz(pi/2) q[8];
ry(pi/2) q[8];
rzz(pi/2) q[1],q[8];
rzz(pi/2) q[1],q[4];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[0];
rz(-pi/2) q[0];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[5];
rz(-pi) q[0];
rz(-pi/2) q[1];
ry(pi/2) q[1];
rz(-pi/2) q[4];
ry(-pi/2) q[4];
ry(-0.025426437503933613) q[5];
rz(pi/2) q[5];
rzz(pi/2) q[6],q[1];
rz(-pi) q[1];
rz(-3*pi) q[8];
rzz(pi/2) q[4],q[8];
rz(0.017777184998498896) q[4];
rzz(pi/2) q[4],q[6];
rzz(pi/2) q[0],q[4];
ry(pi/2) q[0];
ry(-pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[9];
rz(3.123815468591294) q[6];
rzz(pi/2) q[4],q[6];
rz(-pi/2) q[4];
ry(pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[0];
rz(-pi/2) q[0];
ry(-pi/2) q[0];
rz(1.5453698892909635) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[4];
ry(-0.025426437503933277) q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[1],q[6];
ry(-0.025426437503933277) q[1];
rzz(pi/2) q[1],q[4];
ry(-pi/2) q[1];
rzz(pi/2) q[1],q[9];
rz(1.5453698892909635) q[1];
rzz(pi/2) q[1],q[8];
rzz(pi/2) q[0],q[1];
rz(-3*pi/2) q[0];
ry(-pi/2) q[1];
rz(-3*pi/2) q[6];
rzz(pi/2) q[5],q[6];
ry(0.025426437503933044) q[5];
rz(-pi/2) q[5];
rzz(pi/2) q[5],q[4];
rzz(pi/2) q[1],q[4];
rz(-3*pi/2) q[1];
ry(-pi/2) q[4];
rzz(pi/2) q[4],q[6];
rz(-pi) q[4];
ry(-0.025426437503933277) q[5];
rzz(pi/2) q[5],q[6];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[9];
rzz(pi/2) q[3],q[9];
rzz(pi/2) q[0],q[9];
rz(-3*pi/2) q[3];
rz(1.5962227642988296) q[5];
rzz(pi/2) q[5],q[8];
rzz(pi/2) q[1],q[5];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[6];
ry(pi/2) q[6];
rzz(pi/2) q[7],q[8];
rzz(pi/2) q[3],q[8];
rzz(pi/2) q[0],q[8];
rzz(pi/2) q[3],q[5];
ry(pi/2) q[3];
rzz(pi/2) q[2],q[3];
rz(-pi/2) q[3];
ry(-pi/2) q[3];
rzz(pi/2) q[5],q[6];
rzz(pi/2) q[4],q[5];
rzz(pi/2) q[1],q[5];
ry(pi/2) q[1];
ry(pi/2) q[4];
rz(-pi/2) q[6];
ry(pi/2) q[6];
ry(-pi/2) q[8];
rz(-pi) q[8];
