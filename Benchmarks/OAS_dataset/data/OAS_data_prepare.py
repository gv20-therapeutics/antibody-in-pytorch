import json
import os
import glob
import pdb

fnames = glob.glob('/data2/dingqingyang/OAS/json/B*/*.json.gz') #6806
flag_first = True
idx = [('FR1-IMGT',1,26), ('CDR1-IMGT',27,38), 
        ('FR2-IMGT',39,55), ('CDR2-IMGT',56,65), 
        ('FR3-IMGT',66,104), ('CDR3-IMGT',105,117),
        ('FR4-IMGT',118,129)]
cdr_name = {'Heavy': 'h', 'Light': 'l'}
temp_dir = '/data2/dingqingyang/OAS/temp/'
seq_dir = '/data2/dingqingyang/OAS/seq_db/'

for fname in fnames:
    print('---- Processing ' + fname.split('/')[-1] + '-------')
    # decompress the file
    print('decompressing...')
    os.system('gzip -dk %s' % fname)
    json_fname = fname.replace('.gz', '')

    myfile = open(json_fname, 'r')
    data = myfile.readline()
    if len(data) == 0:
        data = myfile.readline()

    seq_fname = json_fname.replace('.json','.txt')
    seq_fname = seq_dir + seq_fname.split('/')[-1]
    # extract meta info from the OAS json file
    #pdb.set_trace()
    obj = json.loads(data)
    keys = []
    values = []
    for key in obj:
        values.append(str(obj[key]))
        keys.append(str(key))

    if flag_first:
        title_line = 'file_name\t' + '\t'.join(keys) + '\t' + 'valid_entry_num' + '\n'
        with open('OAS_meta_info.txt', 'w') as f:
            f.write(title_line)
        flag_first = False

    # extract seq info line by line
    with open(seq_fname, 'w') as f:
        f.write('v\tj\tCDR3_aa\tCounts\tFW1-IMGT\tCDR1-IMGT\tFW2-IMGT\tCDR2-IMGT\tFW3-IMGT\tCDR3-IMGT\tFW4-IMGT\tCDR3-IMGT-111-112\n')
        data = myfile.readline()
        cnt = 1
        valid_cnt = 0

        while data:
            if cnt % 10000 == 0:
                print('line %d' % cnt)
            obj_seq = json.loads(data)
            obj_seq2 = json.loads(obj_seq['data'])

            res = {}
            for key in obj_seq2:
                res.update(obj_seq2[key])
            
            for name, st, ed in idx:
                res[name] = ''
                for k in range(st, ed+1):
                    if str(k) in res:
                        res[name] += res[str(k)]
                        del res[str(k)]
                    else:
                        res[name] += '-'

            remain = [i for i in list(res) if i.startswith('111') or i.startswith('112')]
            res['CDR3-IMGT-111-112'] = ''  # without order??
            for m in remain:
                res['CDR3-IMGT-111-112'] += res[m]
                del res[m]
            #res['CDR3-IMGT-111-112'] = res['CDR3-IMGT-111-112'].str.replace('-','')

            if obj_seq['redundancy'] > 3:
                f.write('\t'.join([obj_seq['v'], obj_seq['j'], obj_seq['cdr3'], str(obj_seq['redundancy']), \
                    res['FR1-IMGT'], res['CDR1-IMGT'], res['FR2-IMGT'], res['CDR2-IMGT'], \
                    res['FR3-IMGT'], res['CDR3-IMGT'], res['FR4-IMGT'], res['CDR3-IMGT-111-112']]) + '\n')
                valid_cnt += 1

            data = myfile.readline()
            cnt += 1

    info_line = seq_fname.split('/')[-1].split('.txt')[0] + \
                '\t' +'\t'.join(values) + '\t' + str(valid_cnt) + '\n'
    with open("OAS_meta_info.txt", "a") as f:
        f.write(info_line)

    myfile.close()
    os.system('rm %s' % json_fname)



