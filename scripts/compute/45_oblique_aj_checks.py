"""45: checks of the oblique-collision geometry.

SM: Sec. S4, "Oblique factor" (the geometric factor of An:2026sdu, per unit area of the collision surface, equals our flux factor).
Prints only; no output files.

 (1) An-Jiang Eq. (5.16) weight vs eta_perp dA_locus for unequal bubbles, by finite differences (independent of the analytic argument);
 (2) 1/sqrt(1 - g'^2) = 1/eta for equal bubbles;
 (3) spacetime overlap of two top-hat sheets of thickness d: d^2/(2 eta) per unit length;
 (4) size of gamma m r for the flat-space benchmark (SM Eq. (S14), beta r = 3, reduced M_P, g_* = 106.75)."""
import numpy as np
# (1) An-Jiang 5.16 weight vs our eta * dA_locus, unequal bubbles, by finite differences (independent of the analytic proof)
for r1, r2 in [(1,1),(1,1.7),(1,4),(0.3,2.5)]:
    s=r1+r2
    for t in [0.05,0.4,2.0,7.0]:
        def xR(t):
            # locus point from geometry: solve |P-O1|=t+r1, |P-O2|=t+r2, O1=(r1,0), O2=(-r2,0)
            x=((t+r2)**2-(t+r1)**2-r2**2+r1**2)/(2*(r1+r2)); R=np.sqrt((t+r1)**2-(x-r1)**2); return x,R
        h=1e-6; (x0,R0),(x1,R1)=xR(t-h),xR(t+h); x,R=xR(t)
        dR=(R1-R0)/(2*h); dx=(x1-x0)/(2*h)
        aj=R*dR/np.sqrt(1+R**2/(r1*r2))               # their 2piR dR weight / (2pi dt)
        u=np.array([x-r1,R])/(t+r1); v=np.array([x+r2,R])/(t+r2)
        eta=0.5*np.linalg.norm(u-v); ours=eta*R*np.hypot(dR,dx)   # eta dA_locus /(2pi dt)
        print(f"r1={r1} r2={r2} t={t}: AJ={aj:.8f} ours={ours:.8f} ratio={aj/ours:.8f}  eta^2*R1R2/(r1r2)={eta**2*(t+r1)*(t+r2)/(r1*r2):.8f}  1-u.v={1-u@v:.6f} 2eta^2={2*eta**2:.6f}")
# (2) equal bubbles: 1/sqrt(1-g'^2) = 1/eta
r=1.0
for R in [0.3,1.0,3.0]:
    gp=R/np.sqrt(r*r+R*R); print("g'",R,1/np.sqrt(1-gp**2), np.sqrt(1+R**2/r**2))
# (3) spacetime overlap of two Gaussian-free top-hat strips of thickness d: integral = A d^2/(2 eta) (2D: per unit length)
d=0.1
for th in [150,90,40,20]:
    n1=np.array([np.sin(np.radians(th/2)),np.cos(np.radians(th/2))]); n2=np.array([-n1[0],n1[1]])
    eta=np.sin(np.radians(th/2))
    # per unit length along the surface (y): integrate over xi (=x) and t at fixed y=0 in the frame; use analytic in t: overlap duration
    xs=np.linspace(-2,2,400001); dur=np.clip(d-np.abs((n1-n2)[0]*xs),0,None)
    print(th, "numeric", np.trapezoid(dur,xs), "d^2/(2eta)", d*d/(2*eta), "slab half-width", d/(2*eta))
# (4) estimate gamma m r for the flat-space benchmark
MP=2.435e18; T=1e10; g=106.75; H=np.sqrt(np.pi**2*g/90)*T**2/MP
for bh in [10,100]:
    beta=bh*H; r=3/beta; gm=10.4*MP/bh
    print(f"beta/H={bh}: H={H:.3g} GeV, gamma m={gm:.3g} GeV, r={r:.3g}/GeV, gamma m r={gm*r:.3g}")
