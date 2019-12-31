'''
Sample data of OAS format

# first line of the file record the basic info of the dataset #
{"Longitudinal": "no", "Chain": "Light", "Author": "Vander Heiden et al., (2017)", "Isotype": "Bulk", "Age": "31", "Size_igblastn": 944876, "Disease": "None", "Link": "https://www.ncbi.nlm.nih.gov/pubmed/28087666", "BSource": "PBMC", "BType": "Unsorted-B-Cells", "Subject": "HD10", "Species": "human", "Vaccine": "None", "Size": 619068}

# each entry stores the name and sequence of the amino acid (optional: indexing of each residue and domain; v/d/j gene identified by blast) #
{"num_errors": "2", "redundancy": 1, "name": 1, "seq": "DVVMTQSPLSLPVTLGQPASISCRSSQSLVYADGNTYLNWFQQRPGQSPRRLIYKVSNRDSGVPDRFSGSGSGTNFTLKISRVEAEDVGVYYCMQGTHWPWTFGQGTKVEIK", "v": "IGKV2-30*01", "cdr3": "MQGTHWPWT", "original_name": "728536", "errors": "[(u'32', u'A'), (u'86', u'N')]", "j": "IGKJ1*01", "data": "{\"cdrl1\": {\"27\": \"Q\", \"32\": \"A\", \"31\": \"Y\", \"30\": \"V\", \"28\": \"S\", \"29\": \"L\", \"35\": \"G\", \"34\": \"D\", \"38\": \"Y\", \"37\": \"T\", \"36\": \"N\"}, \"cdrl2\": {\"57\": \"V\", \"56\": \"K\", \"65\": \"S\"}, \"cdrl3\": {\"108\": \"T\", \"115\": \"P\", \"114\": \"W\", \"117\": \"T\", \"109\": \"H\", \"116\": \"W\", \"106\": \"Q\", \"107\": \"G\", \"105\": \"M\"}, \"fwl4\": {\"120\": \"Q\", \"121\": \"G\", \"122\": \"T\", \"123\": \"K\", \"124\": \"V\", \"125\": \"E\", \"126\": \"I\", \"127\": \"K\", \"119\": \"G\", \"118\": \"F\"}, \"fwl1\": {\"24\": \"R\", \"25\": \"S\", \"26\": \"S\", \"20\": \"S\", \"21\": \"I\", \"22\": \"S\", \"23\": \"C\", \"1\": \"D\", \"3\": \"V\", \"2\": \"V\", \"5\": \"T\", \"4\": \"M\", \"7\": \"S\", \"6\": \"Q\", \"9\": \"L\", \"8\": \"P\", \"11\": \"L\", \"10\": \"S\", \"13\": \"V\", \"12\": \"P\", \"15\": \"L\", \"14\": \"T\", \"17\": \"Q\", \"16\": \"G\", \"19\": \"A\", \"18\": \"P\"}, \"fwl3\": {\"91\": \"I\", \"88\": \"T\", \"89\": \"L\", \"66\": \"N\", \"67\": \"R\", \"68\": \"D\", \"83\": \"S\", \"80\": \"G\", \"86\": \"N\", \"87\": \"F\", \"84\": \"G\", \"85\": \"T\", \"92\": \"S\", \"79\": \"S\", \"69\": \"S\", \"104\": \"C\", \"78\": \"G\", \"77\": \"S\", \"76\": \"F\", \"75\": \"R\", \"74\": \"D\", \"72\": \"P\", \"71\": \"V\", \"70\": \"G\", \"102\": \"Y\", \"90\": \"K\", \"100\": \"G\", \"101\": \"V\", \"95\": \"E\", \"94\": \"V\", \"97\": \"E\", \"96\": \"A\", \"99\": \"V\", \"98\": \"D\", \"93\": \"R\", \"103\": \"Y\"}, \"fwl2\": {\"52\": \"R\", \"39\": \"L\", \"48\": \"Q\", \"49\": \"S\", \"46\": \"P\", \"47\": \"G\", \"44\": \"Q\", \"45\": \"R\", \"51\": \"R\", \"43\": \"Q\", \"40\": \"N\", \"42\": \"F\", \"55\": \"Y\", \"53\": \"L\", \"54\": \"I\", \"41\": \"W\", \"50\": \"P\"}}"}

'''

class OAS_format_seq():
	# takes the entry of OAS format and extract sequence information as required
    seq_name = ''
    seq_dict = {}
    chain_type = ''
    seq_category = ''
    seq_length = 0
    def __init__(self, raw_info_sect):
        raw_list = raw_info_sect.strip('\n').split('\n')
        N_tot = len(raw_list)
        self.seq_length = N_tot - 7
        self.seq_name = raw_list[0].split('\t')[0].split(':')[1]
        self.seq_category = raw_list[0].split('\t')[1]
        self.chain_type = raw_list[5].split('|')[2]
        
        seq_dict_store = {}
        for i in range(7,N_tot):
            pos_list = [item for item in raw_list[i].split(' ') if len(item) > 0]
            for j in range(2,len(pos_list)):
                if not pos_list[j] == '-':
                    if pos_list[1] in seq_dict_store:
                        seq_dict_store[pos_list[1]].append(pos_list[j])
                    else:
                        seq_dict_store[pos_list[1]] = [pos_list[j]]
        self.seq_dict = seq_dict_store
    
    def calculate_identity(self, seq_dict2):
        count = 0
        tot_len = 0
        for key in self.seq_dict.keys():
            if key in seq_dict2:
                for xx in seq_dict2[key]:
                    if xx in self.seq_dict[key]:
                        count = count + 1.
        return count / self.seq_length

    def ANARCI_indexing():
    	pass
#------------------------------------------------------------------
'''
Sample data of plain text format

1. (gapped, full length) -QVQLVQS-GAEVKKPGSSVKVSCTTSG-GTFSS-----FVINWMRQAPGQGLGWRGGIMPV---FDTANFAQKVQGRVTMTADESKRTIYMERSSLRSEETAVYYCARSVP------------------------VKD-----------
2. (without gap, full length) QVQLVQSGAEVKKPGSSVKVSCTTSGGTFSSFVINWMRQAPGQGLGWRGGIMPVFDTANFAQKVQGRVTMTADESKRTIYMERSSLRSEETAVYYCARSVPVKD
3. (CDRH3 only) QQYNSFPLT

'''

def plain2OAS(plain_text, output_fn, gapped = False, seqType = 'AA'):
	pass

#------------------------------------------------------------------
'''
An inhereted data structure from pytorch, which would takes in a plain text or OAS format dataset, using OAS_format_seq or plain2OAS utility functions to extract required sequences/values as input/output.
'''

from torch.utils.data import Dataset

aa_store = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
aa2id = dict(zip(aa_store,  list(range(len(aa_store)))))
id2aa = dict(zip(list(range(len(aa_store))), aa_store))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]

class ProteinSeqDataset(Dataset):
    def __init__(self, fn, fo):
        # initialize file path or list of file names
        with open(fn, 'r') as f:
            self.data = [l.strip('\n') for l in f]
        self.X = [[one_of_k_encoding(aa, aa_store) for aa in r] for r in self.data]

        with open(fo, 'r') as f:
            self.label = [l.strip('\n') for l in f]

        assert len(self.X) == len(self.label)

    def __getitem__(self,index):
        # 1. read one data from file
        # 2. preprocess the data
        # 3. return a data pair (X and y)
        return self.X[idx], self.label[idx]

    def __len__(self):
        # return the total size of the dataset
        return len(self.X)

