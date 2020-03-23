import codecs

from sklearn_crfsuite import metrics

MODE = 11

GOLD = '../plain/ctrip_test_states.utf8'

if MODE == 1:
    TEST = 'ctrip_test_B20-E60-F10-PU-Bi-RCT-CT-Bn-De_states.txt'

if MODE==2:
    TEST='ctrip_test_B20-E60-F5-PU-Bi-Bn-De_states.txt'

if MODE==3:
    TEST = 'ctrip_test_B20-E60-F5-PU-C-Bi-Bn-De_states.txt'

if MODE==4:
    TEST = 'ctrip_test_B20-E60-F5-PU-C-Bi-Bn-De_states.txt'

if MODE==5:
    TEST='ctrip_test_B20-E60-F5-PU-Bi-C-Bn-De_states.txt'

if MODE == 6:
    TEST="ctrip_test_B20-E60-F5-PU-C64-K1-Bi-Bn-De_states.txt"

if MODE ==7:
    TEST="ctrip_test_B20-E60-F3-PU-C64-K1-Bi-Bn-De_states.txt"

if MODE==8:
    TEST="ctrip_test_B20-E100-F3-PU-C150-K3-Bi-Bn-De_states.txt"

if MODE==9:
    TEST="ctrip_test_B64-E200-F3-PU-C150-Mx-K3-Bi-Bn-De_states.txt"

if MODE==10:
    TEST="ctrip_test_B64-E200-F3-PU-Bi-A-Bi-Mx-Bn-De_states.txt"

if MODE == 11:
    TEST = 'ctrip_test_B20-E60-F10-RU-Bi-RCT-CT-Bn-De_states.txt'

with codecs.open(TEST, 'r', encoding='utf8') as fj:
    with codecs.open(GOLD, 'r', encoding='utf8') as fg:
        jstates = fj.readlines()
        states = fg.readlines()
        y = []
        for state in states:
            state = state.strip()
            y.append(list(state))
        yp = []
        for jstate in jstates:
            jstate = jstate.strip()
            yp.append(list(jstate))
        # for i in range(len(y)):
        assert len(yp) == len(y)
        m = metrics.flat_classification_report(
            y, yp, labels=list("PN"), digits=4
        )
        print(m)
