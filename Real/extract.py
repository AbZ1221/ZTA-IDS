import sys

filename = sys.argv[1]

lines = open(filename+".csv").readlines()
nlines = [lines[0].strip() + ',trans_depth']
for line in lines[1:]:
    line = line.strip()
    if line.split(",")[5] == 'http':
        line += ',1'
    else:
        line += ',0'
    nlines.append(line)

open(filename+"_tp.csv", "w").write("\n".join(nlines))

lines = open(filename+"_tp.csv").readlines()

nlines = [lines[0].strip() + ",ct_srv_src"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[0] == line_.split(",")[0] and line.split(",")[5] == line_.split(",")[5]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctsrvs.csv", "w").write("\n".join(nlines))
"ct_srv_src" 

lines = open(filename+"_ctsrvs.csv").readlines()

nlines = [lines[0].strip() + ",ct_state_ttl"]
for line in lines[1:]:
    state = line.split(",")[6]
    try:
        sttl = int(line.split(",")[11])
    except:
        sttl = 0
    try:
        dttl = int(line.split(",")[12])
    except:
        dttl = 0
    if state == 'FIN':
        if dttl > 251 and (sttl > 250 or (sttl < 75 and sttl > 50)):
            ct_state_ttl = 1
        else:
            ct_state_ttl = 0
    elif state == 'CON':
        if sttl > 61 and sttl < 63 and dttl > 250 and dttl < 255:
            ct_state_ttl = 3
        else:
            ct_state_ttl = 0
    elif state == 'REQ':
        if sttl > 250 and sttl < 255:
            ct_state_ttl = 6
        else:
            ct_state_ttl = 0
    else:
        ct_state_ttl = 0
    nlines.append(line.strip() + ',' + str(ct_state_ttl))

open(filename+"_ctstate.csv", "w").write("\n".join(nlines))


lines = open(filename+"_ctstate.csv").readlines()

nlines = [lines[0].strip() + ",ct_dst_ltm"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[1] == line_.split(",")[1]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctdst.csv", "w").write("\n".join(nlines))


lines = open(filename+"_ctdst.csv").readlines()

nlines = [lines[0].strip() + ",ct_src_dport_ltm"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[0] == line_.split(",")[0] and line.split(",")[5] == line_.split(",")[5]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctdp.csv", "w").write("\n".join(nlines))



lines = open(filename+"_ctdp.csv").readlines()

nlines = [lines[0].strip() + ",ct_dst_sport_ltm"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[1] == line_.split(",")[1] and line.split(",")[2] == line_.split(",")[2]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctdss.csv", "w").write("\n".join(nlines))



lines = open(filename+"_ctdss.csv").readlines()

nlines = [lines[0].strip() + ",ct_dst_src_ltm"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[0] == line_.split(",")[0] and line.split(",")[1] == line_.split(",")[1]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctds.csv", "w").write("\n".join(nlines))


lines = open(filename+"_ctds.csv").readlines()

nlines = [lines[0].strip() + ",ct_src_ltm"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[0] == line_.split(",")[0]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctsi.csv", "w").write("\n".join(nlines))


lines = open(filename+"_ctsi.csv").readlines()

nlines = [lines[0].strip() + ",ct_srv_dst"]
for i, line in enumerate(lines[1:]):
    cnt = 0
    for i_, line_ in enumerate(lines[max(1, i-100):i]):
        if line.split(",")[1] == line_.split(",")[1] and line.split(",")[5] == line_.split(",")[5]:
            cnt += 1
    line = line.strip() + ',' + str(cnt)
    nlines.append(line)

open(filename+"_ctsrvd.csv", "w").write("\n".join(nlines))

mappings={
    '80': "http",
    '20': "ftp",
    '21': "ftp",
    '22': "ssh",
    '25': "smtp",
    '53': "dns",
    '20': "ftp-data",
    '194': "irc",
    '113': "irc"
}

lines = open(filename+"_ctsrvd.csv").readlines()
nlines = [",".join(['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
       'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
       'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
       'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
       'ct_src_ltm', 'ct_srv_dst'])+"\n"]
for i, line in enumerate(lines[1:]):
    parts = line.split(",")[3:]
    if parts[2] in mappings.keys():
        line = ",".join(parts[:2]) + "," + mappings[parts[2]] + "," + ",".join(parts[3:])
    else:
        line = ",".join(parts[:2]) + ",-," + ",".join(parts[3:])
    nlines.append(str(i) + "," + line)

open(filename+"_test.csv", "w").write("".join(nlines))



lines = open(filename+".csv").readlines()
matches = []
for i, line in enumerate(lines[1:]):
    line = line.strip()
    if line.split(",")[1] == "5.161.141.211":
        matches.append(i)

print(matches)
