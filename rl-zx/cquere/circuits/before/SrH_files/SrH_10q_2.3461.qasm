OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
ry(-pi/2) q[0];
ry(-pi/2) q[1];
rz(-7*pi/2) q[4];
rz(-3*pi/2) q[5];
rz(-3*pi/2) q[6];
ry(pi/2) q[9];
rx(pi) q[9];
rzz(pi/2) q[6],q[9];
ry(pi/2) q[6];
rz(-pi/2) q[9];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[6];
rz(-pi/2) q[6];
ry(-pi/2) q[6];
rz(-pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[6],q[9];
rz(-pi/2) q[6];
ry(pi/2) q[6];
rz(-pi) q[9];
rzz(pi/2) q[5],q[9];
ry(pi/2) q[5];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[5];
rz(-pi/2) q[5];
ry(-pi/2) q[5];
rz(-pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[5],q[9];
rzz(pi/2) q[1],q[5];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rz(-pi/2) q[9];
rzz(pi/2) q[4],q[9];
rzz(pi/2) q[4],q[8];
rzz(pi/2) q[4],q[7];
rzz(pi/2) q[4],q[6];
rzz(pi/2) q[3],q[4];
rzz(pi/2) q[2],q[4];
rzz(pi/2) q[1],q[4];
rzz(pi/2) q[0],q[4];
rz(-pi) q[0];
rz(-pi/2) q[1];
ry(pi/2) q[1];
ry(-pi/2) q[4];
rz(-3.11653260102098) q[4];
rzz(pi/2) q[0],q[4];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[1];
rz(-9.449838013338193) q[0];
rzz(pi/2) q[0],q[9];
rz(-3*pi/2) q[1];
rzz(pi/2) q[4],q[6];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rz(-3.11653260102098) q[6];
rzz(pi/2) q[0],q[6];
ry(-pi/2) q[6];
rz(-3*pi) q[7];
rz(-3*pi) q[8];
rzz(pi/2) q[5],q[8];
rzz(pi/2) q[5],q[7];
ry(-pi/2) q[5];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[1];
rz(0.025060052568813163) q[9];
rzz(pi/2) q[6],q[9];
rz(pi) q[6];
ry(pi/2) q[9];
rzz(pi/2) q[4],q[9];
rzz(pi/2) q[0],q[4];
rzz(pi/2) q[0],q[9];
rzz(pi/2) q[0],q[6];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
ry(-pi/2) q[0];
rz(pi) q[4];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rz(-pi/2) q[5];
ry(pi/2) q[5];
rz(-1.5958563793637097) q[6];
rzz(pi/2) q[4],q[6];
rzz(pi/2) q[0],q[6];
ry(-pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[1];
ry(-pi/2) q[1];
rz(1.5374939667946972) q[6];
ry(pi/2) q[9];
rzz(pi/2) q[0],q[9];
ry(-pi/2) q[0];
rz(1.5457362742260834) q[9];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[5];
rzz(pi/2) q[4],q[9];
rzz(pi/2) q[0],q[9];
rz(-9.44983801333819) q[0];
rz(-7.82892158140567) q[4];
rzz(pi/2) q[4],q[8];
rzz(pi/2) q[0],q[8];
rzz(pi/2) q[4],q[7];
rzz(pi/2) q[0],q[7];
rzz(pi/2) q[3],q[4];
rzz(pi/2) q[2],q[4];
rzz(pi/2) q[1],q[4];
rz(-0.03330236000019848) q[1];
ry(pi/2) q[4];
rz(pi/2) q[5];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
rzz(pi/2) q[0],q[4];
rzz(pi/2) q[0],q[3];
rzz(pi/2) q[0],q[2];
rzz(pi/2) q[0],q[1];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[6];
rz(-1.5374939667946972) q[0];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[6];
rz(pi/2) q[0];
ry(pi) q[0];
ry(-pi/2) q[1];
rz(pi/2) q[4];
ry(pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[4];
rz(-9*pi/2) q[4];
rz(pi/2) q[6];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[9];
rz(-1.5374939667946972) q[6];
rzz(pi/2) q[6],q[8];
rzz(pi/2) q[6],q[7];
rzz(pi/2) q[0],q[6];
rz(pi/2) q[0];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[9];
rz(-6.316487667179786) q[0];
rzz(pi/2) q[0],q[8];
rzz(pi/2) q[0],q[7];
ry(pi/2) q[6];
rzz(pi/2) q[0],q[6];
rzz(pi/2) q[0],q[5];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[4];
rz(1.550870781794143) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[5];
rz(pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[0],q[6];
rz(-pi/2) q[0];
rzz(pi/2) q[0],q[4];
ry(0.019925545000753747) q[0];
rz(-pi/2) q[0];
rzz(pi/2) q[0],q[6];
ry(-0.019925545000754108) q[0];
rz(-pi) q[0];
rzz(pi/2) q[0],q[6];
rzz(pi/2) q[0],q[4];
ry(1.550870781794142) q[0];
rz(-pi) q[0];
rzz(pi/2) q[1],q[6];
ry(-0.03330236000019921) q[1];
rz(-pi/2) q[1];
rzz(pi/2) q[1],q[4];
ry(-3.108290293589594) q[1];
rz(-pi/2) q[1];
rzz(pi/2) q[1],q[6];
ry(3.108290293589594) q[1];
rz(-3*pi/2) q[1];
rzz(pi/2) q[1],q[4];
rzz(pi/2) q[1],q[5];
ry(pi/2) q[1];
ry(1.59072187179565) q[5];
rz(pi/2) q[5];
rzz(pi/2) q[5],q[4];
ry(0.019925545000753747) q[5];
rz(-pi/2) q[5];
rzz(pi/2) q[5],q[6];
ry(-0.019925545000754108) q[5];
rz(-pi) q[5];
rzz(pi/2) q[5],q[4];
rzz(pi/2) q[0],q[4];
ry(-pi/2) q[4];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
rz(-1.5907218717956502) q[5];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[1];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[3],q[5];
rzz(pi/2) q[2],q[5];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rzz(pi/2) q[0],q[5];
ry(-pi/2) q[0];
rzz(pi/2) q[4],q[5];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rz(pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[1],q[6];
rz(pi) q[1];
rz(-pi/2) q[6];
rzz(pi/2) q[0],q[6];
rz(-pi) q[0];
rzz(pi/2) q[0],q[5];
rzz(pi/2) q[0],q[1];
ry(pi/2) q[1];
rx(-pi/2) q[9];
