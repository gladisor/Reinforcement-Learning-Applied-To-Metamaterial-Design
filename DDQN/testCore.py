import matlab.engine
import torch

eng = matlab.engine.start_matlab()
eng.addpath('TSCS')

xM = matlab.double([
	[0.0, 0.0]])

av = matlab.double([
	[1.0]])

c_pv = matlab.double(
	[5480.0]*1)

rho_shv = matlab.double(
	[8850.0]*1)

print(c_pv)

print(rho_shv)

k0amax = matlab.double([0.45])
k0amin = matlab.double([0.35])
nfreq = matlab.double([11])

tscs = eng.getMetric_thinShells_radii_material(xM, av, c_pv, rho_shv, k0amax, k0amin, nfreq)