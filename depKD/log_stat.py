import os

res = {}
wrong = []
refine = []
bath_path = 'logs/tune'
files = os.listdir(bath_path)
for file_name in files:
    with open(os.path.join(bath_path, file_name), 'r') as fr:
        lines = fr.readlines()
        try:
            # score = (lines[-14].split('-')[-1]).split(' ')[-1].strip()
            score = (lines[-5].split('\t')[-1]).strip()

            try:
                score = float(score)
            except:
                refine.append(file_name)
                continue
        except:
            wrong.append(file_name)
            continue
        res[file_name.rstrip('.yaml.log')] = score
        # print(score)
        # exit()
# refine2 = []
# for file_name in refine:
#     with open(os.path.join(bath_path, file_name), 'r') as fr:
#         lines = fr.readlines()
#         try:
#             # score = (lines[-14].split('-')[-1]).split(' ')[-1].strip()
#             score = (lines[-6].split('\t')[-1]).strip()
#             try:
#                 score = float(score)
#             except:
#                 refine.append(file_name)
#                 continue
#         except:
#             refine2.append(file_name)
#             continue
#         res[file_name.rstrip('.yaml.log')] = score
print(refine)
# print(refine2)
print(wrong)
res_print = sorted(res.items(), key=lambda x:x[1], reverse=True)

with open('aaaastatistic', 'w') as fw:
    for i in res_print:
        fw.write(i[0]+'\t'+str(i[1])+'\n')