from tastytrade import ProductionSession, CertificationSession, DXLinkStreamer
from tastytrade.dxfeed import EventType

s, username, password = input("Production (p) or Certification (c), username & password").strip().split()
if s == 'p':
    try:
        session = ProductionSession(username,password)
    except:
        print("sum ting wong (failed to login to prod)")
elif s == 'c':
    try:
        session = CertificationSession(username,password)
    except:
        print("sum ting wong (failed to login to cert)")
else:
    raise ValueError('sum ting wong (p/c)')


