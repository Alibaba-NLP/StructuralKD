
T=1; F=0
B=0;I=1;O=2;E=3;S=4
# R = self.tag_dictionary.idx2item
# rel_scores: sent_len * sent_len * n_roles
role2id = self.tag_dictionary.item2idx
Roles = [b'None', b'ORG', b'MISC', b'LOC', b'PER']
prefixes = [B,I,E,S]    # 'O' is not prefix
n = sent_len    
DPF = {}
DPB = {}

#initialization
for i in range(1,n+1):
    DPF[(i,F)] = 0
    DPB[(i,F)] = 0
    for role in Roles:
        DPF[(i,role)] = 0
        DPB[(i,role)] = 0
    

# base case
for role in Roles:
    DPF[(0, role)] = 1
    DPB[(n+1, role)] = 1
DPF[(0, F)] = 1
DPB[(n+1, F)] = 1

#define rules for DPF(m, F) and DPB(m, F)
# DPF(m, R) = exp(S(1,m,R))*DPF(1,F) + exp(S(2,m,R))*DPF(2,F) + ... + exp(S(m,m,R))*DPF(m,F)
# DPF(m, F) = DPF(m-1, F) + DPF(m-1, role1) + DPF(m-2, role2) + ... 
 

rules_F = {}
rules_B = {}
for m in range(1,n+1):
    #DPF(m,F), DPB(m,F)
    rule_f = [(m-1,F)]
    rule_b = [(m+1, F)]
    for role in Roles:
        rule_f.append((m-1, role))
        rule_b.append((m+1, role))
    rules_F[(m,F)] = rule_f
    rules_B[(m,F)] = rule_b
    #DPF(m,R), DPB(m,R)
    rule_fr = []
    rule_br = []
    for i in range(1, m+1):
        rule_fr.append((i, F))
    for j in range(m, n+1):
        rule_br.append((j, F))
    for role in Roles:
        rules_F[(m, role)] = rule_fr
        rules_B[(m, role)] = rule_br
    

# update DPF, DPB
for m in range(1,n+1):
    for l_entry in rules_F: # l_entry: (m, F) or (m, role)
        score = 0
        r_entries = rules_F[l_entry]
        _, tag = l_entry
        if tag in {0, '0'}: #DPF(m,F)
            for r_entry in r_entries:
                score += DPF[r_entry]
            DPF[l_entry] = score
        else:               #DPF(m, R)
            for r_entry in r_entries:
                i, _ = r_entry
                score_r = rel_scores[i, m, role2id[tag]]
                score += math.exp(score_r)*DPF[(i,F)]
            DPF[l_entry] = score
        
for m in range(m, 0, -1):
    for l_entry in rules_B:
        score = 0
        r_entries = rules_B[l_entry]
        _, tag = l_entry
        if tag in {0, '0'}:
            for r_entry in r_entries:
                score += DPB[r_entry]
            DPB[l_entry] = score
        else:               #DPF(m, R)
            for r_entry in r_entries:
                j, _ = r_entry
                score_r = rel_scores[m, j, role2id[tag]]
                score += math.exp(score_r)*DPF[(j,F)]
            DPB[l_entry] = score




#------------------------------------
# caculate scores for BIOES tags in sequence model
DPC = {}
#initialization
for m in range(1,n+1):
    DPC[(O, m)] = 0
    for role in Roles:
        for prefix in [B,I,E,S]:
            DPC[(prefix, role, m)] = 0
            
score_ij = _score_ij(rel_scores, DPF, DPB)
DPC = _DPI(score_ij, DPC)

for m in range(1, n+1):
    DPC[(O, m)] = DPF[(m,F)]*DPB[(m,F)]
    for role in Roles:
        DPC[(B, role, m)] = DPF[(m, F)]*DPB[(m, role)]
        DPC[(E, role, m)] = DPF[(m, role)]*DPB[(m,F)]
        score_s = rel_scores[m, m, role2id[role]]
        DPC[(S, role, m)] = DPF[(m, F)]*math.exp(score_s)*DPB[(m, F)]



#calculate for prefix 'I'
def _score_ij(rel_scores, DPF, DPB):
    for role in Roles:
        score_ij = {}
        for i in range(1, n-1):
            for j in range(2, n+1):
                score_ij[(i,j,role)]= math.exp(rel_scores[i, j, role2id[role]]) * DPF[(i,F)]*DPB[(j,F)]
    return score_ij

def _DPI(score_ij, DPC):
    #intial
    Cum = {}
    for role in Roles:
        Cum[(1,1,role)] = 0
        Cum[(n,n,role)] = 0
        for i in range(1,n-1):
            for j in range(i-1, i+1):
                #calculate Cum(i+1,j,R)
                Cum[(i+1, j, role)] = Cum[(i,j,role)]
                for v in range(j+1, n+1):
                    Cum[(i+1, j, role)] += score_ij[(i, v, role)]
                #calculate Cum(i+1,j+1,R)
                Cum[(i+1,j+1, role)] = Cum[(i+1,j, role)]
                for u in range(1,i+1):
                    Cum[(i+1,j+1, role)] += score_ij[(u,j+1,role)]
        
        # socre of tags with prefix 'I'
        for m in range(1,n+1):
            DPC[(I, role, m)] = Cum[(m, m, role)]
        
    return DPC






