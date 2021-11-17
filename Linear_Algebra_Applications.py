# ----------------------------Decryption Problem-----------------------------
# decrypt S into character values
S=["10101","00100","10101","01011","00011","01011","10101","00100","11001","11010"]
A=[x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ "]
D={A[num]:num for num in range(27)}

#converting numbers to binary up to 0Xb (X)Bits
binNums=[f'{num:05b}' for num in range(27)]
BinD={binNums[x]:A[x] for x in range(27)}

message=[BinD[s] for s in S]
print(message)

# ------------------------------Mini Search Engine---------------------------------------
# creates a dictionary for each word in given document/txt file
def invind(doclist):
    ld = list(doclist)
    d = {}
    for doc in ld:
        splitdoc=doc.split()
        for word in splitdoc:
            d[word] = {numdoc for numdoc in range(len(ld)) if word in ld[numdoc]}
    return d

# Returns sentences/docs that contain any of the words given
def searchOr(InvIndex, Query):
       return [InvIndex[words] for words in Query]

#print(searchOr(invind(doc1),['gas','lame']))

# Returns sentences/docs that contain both words given
def searchAnd(InvIndex, Query):
    return set(sum([list(InvIndex[words]) for words in Query],[]))

print(searchAnd(invind(doc1), ['gas', 'lame']))

# ------------------Application of Dot Product in Determining Agreements/Disagreements --------------------
# Lab and .txt file taken from CODING THE MATRIX BOOK

voting_file = list(open("US_Senate_voting_data_109.txt"))

def voting_dict(voting_data):
    d = {}
    for senator in voting_data:
        senator = senator.split()
        d[senator[0]] = [int(vt) for vt in senator[3:len(senator)]]
    return d


def policy_comp(senA, senB, func_voting_dict):
    return sum([func_voting_dict[senA][i]*func_voting_dict[senB][i] for i in range(len(func_voting_dict[senA]))])

# print(policy_comp("Akaka", "Alexander", voting_dict(list(voting_file))))

def similiairity_comparision(sen, votes):
    agreements = []
    for senator in votes:
        if senator != sen:
            agreements.append(policy_comp(sen, senator, votes))

    agreement_score = max(agreements)
    disagreement_score = min(agreements)

    for score in agreements:
        if score == agreement_score:
            sen_name_agree = agreements.index(agreement_score)
        elif score == disagreement_score:
            sen_name_disagree = agreements.index(disagreement_score)

    return [list(votes.keys())[sen_name_disagree], list(votes.keys())[sen_name_agree]]


# print(similarity_comparison("Chafee",voting_dict(voting_file)))

def state_agreement(state,vote_data):
    d = []
    for senator in vote_data:
        senator = senator.split()
        if senator[2] == state:
            d.append([int(vt) for vt in senator[3:len(senator)]])

    return sum([d[0][i]*d[1][i] for i in range(len(d[0]))])

# print(state_agreement("CA", voting_file))


def average_agreement(senator, senator_list, senator_voting_record):
    agreements = []
    for name in senator_list:
        if senator != name:
            agreements.append(policy_comp(senator, name, senator_voting_record))
    return sum(agreements)/len(agreements)


#print(average_agreement("Chafee",["Kyl","Kohl","Kerry"],voting_dict(voting_file)))

def average_record(senators, senator_voting_record):
    l= sum([np.asarray(senator_voting_record[name]) for name in senators])/len(senators)
    return l

#print(average_record(["Kyl","Kohl","Kerry"],voting_dict(voting_file)))

dems ={}
for rows in voting_file:
    rows = rows.split()
    if rows[1] == "D":
        dems[rows[0]] = [int(vt) for vt in rows[3:len(rows)]]

average_dems = average_record(list(dems.keys()), voting_dict(voting_file))
#print(average_dems)

#---------------------------Plotting Line Segment Between Any Two Given Points---------------------
def seg(p1,p2,step):
    slope = float((p2[1]-p1[1])/(p2[0]-p1[0]))
    y_int = float(p1[1]-slope*p1[0])
    
    if p1[0]<p2[0]:
        line_seg = [float(slope*x+y_int) for x in np.arange(p1[0], p2[0],step)]
    else:
        line_seg = [(x, slope * x + y_int) for x in np.arange(p2[0], p1[0],step)]

	plt.pyplot.scatter([x for (x,y) in line_seg],[y for (x,y) in line_seg])
	plt.pyplot.show()








