import json
import os
import glob
import pdb

fnames = glob.glob('/data2/dingqingyang/OAS/json/*/*.json.gz')
flag_first = True
ref_cdr = [str(xx) for xx in range(105, 118)]
ref_full_length = [str(xx) for xx in range(1,151)]
cdr_name = {'Heavy': 'cdrh3', 'Light': 'cdrl3'}

for fname in fnames[0:20]:
    print('Processing ' + fname)
    os.system('gzip -dk %s' % fname)

    pdb.set_trace()
    json_fname = fname.replace('.gz', '')
    with open(json_fname, 'r') as myfile:
        data = myfile.read()
    data_list = data.strip('\n').split('\n')

    # extract meta info from the OAS json file
    obj = json.loads(data_list[0])
    keys = []
    values = []
    for key in obj:
        values.append(str(obj[key]))
        keys.append(str(key))
    info_line = fname + '\t' +'\t'.join(values) + '\n'

    if flag_first:
        title_line = 'file_name\t' + '\t'.join(keys) + '\n'
        with open('OAS_meta_info.txt', 'w') as f:
            f.write(title_line)
        flag_first = False
    with open("OAS_meta_info.txt", "a") as f:
        f.write(info_line)

    seq_fname = json_fname.replace('.json','.txt')
    seq_fname = '/data2/dingqingyang/OAS/cdr3_seq/' + seq_fname.split('/')[-1]

    with open(seq_fname, 'w') as f:
        f.write('v\tj\toriginal_name\tcdr3\tgapped_cdr\tfull_length_seq\tgapped_full\n')
        for m in range(1, len(data_list)):
            # extract cdr seq line-by-line
            obj_seq = json.loads(data_list[m])
            obj_seq2 = json.loads(obj_seq['data'])
            gapped_seq = []
            for kk in ref_cdr:
                if kk in obj_seq2[cdr_name[obj['Chain']]]:
                    gapped_seq.append(obj_seq2[cdr_name[obj['Chain']]][kk])
                else:
                    gapped_seq.append('-')
            gapped_seq = ''.join(gapped_seq)

            # extract full length seq line-by-line
            new_dict = {}
            for key in obj_seq2:
                new_dict.update(obj_seq2[key])
            gapped_full = []
            for kk in ref_full_length:
                if kk in new_dict:
                    gapped_full.append(new_dict[kk])
                else:
                    gapped_full.append('-')
            gapped_full = ''.join(gapped_full)

            f.write('\t'.join([obj_seq['v'], obj_seq['j'], obj_seq['original_name'], 
                        obj_seq['cdr3'], gapped_seq, obj_seq['seq'], gapped_full]) + '\n')

    os.system('rm %s' % json_fname)





