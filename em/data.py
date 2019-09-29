from importing import *
from sparse import *
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def users_to_df(fname,names):
    # read followers' network file,each entry is: username1, username2, 1.0
    n0 = str(names[0])
    n1 = str(names[1])

    users_content = read_csv(fname, names=names)
    users_content[n0] = to_numeric(users_content[n0])
    users_content[n1] = to_numeric(users_content[n1])
    users_d = {}
    num = 0

    for u in users_content[n0]:
        if u not in users_d:
            users_d[u] = num
            num += 1

    for u in users_content[n1]:
        if u not in users_d:
            users_d[u] = num
            num += 1

    for ind, row in users_content.iterrows():
        users_content.at[ind, n0] = users_d[row[n0]]
        users_content.at[ind, n1] = users_d[row[n1]]

    return users_content

def prop_pi():

    #pi0= np.random.uniform(0, 1, 1)[0]                  #probability that a tweet belongs to one of the k=2 categories

    #pi0 = 1e-8
    pi0 = 0.5
    pi1 = 1 - pi0

    return np.array([pi0,pi1])

def assign_user_phi(D, b, option):                      #phi(uk)= the probability that user u initiates a tweet of type k
                                                        #sum of phi(uk)'s for a fixed category k over all users is 1
                                                        #Sum(phi0)=1, Sum(phi1)=1 for all users

    if option =='m':                                    #marginal case
        #print ("assigning phis: all zero but one")
        phi0 = np.zeros(D, dtype=np.float64)
        phi1 = np.zeros(D, dtype=np.float64)
        phi0[0] = 1
        phi1[0] = 1

    elif option == 'r':                     #uniformly random assignment case

        #print ("assigning phi: uniformly at random (between 1 and D)")

        l1 = np.random.uniform(0, 1, D)                #very small number instead of 0
        phi0 = l1 / l1.sum()
        l2 = np.asarray([min(b * l1[i], 1) for i in range(D)])
        phi1 = l2 / l2.sum()


    elif option == 'u':                     #uniform case
        #print ("assigning phi: phi = 1/D for all")
        phi0 = np.full(shape=D, fill_value=1. / D, dtype=np.float64)
        phi1 = np.full(shape=D, fill_value=1. / D, dtype=np.float64)

    log_phi0 = np.log(phi0, dtype=np.float128)
    log_phi1 = np.log(phi1, dtype=np.float128)

    log_phi0 = np.nan_to_num(log_phi0)
    log_phi1 = np.nan_to_num(log_phi1)

    return phi0, phi1, log_phi0, log_phi1

def assign_user_gamma(D, a):

    for i in range(D):
        n1 = np.random.uniform(0, 1)
        temp = min(a * n1, 1)
        gamma0[i] = n1
        gamma1[i] = temp
        tmp1 = 1-n1
        tmp2 = 1-temp

        log_gamma0[i] = np.log(n1, dtype=np.float128)
        log_1_gamma0[i] = np.log(tmp1, dtype=np.float128)
        log_gamma1[i] = np.log(temp, dtype=np.float128)
        log_1_gamma0[i] = np.log(tmp2, dtype=np.float128)

        log_gamma0[i] = np.nan_to_num(log_gamma0[i])
        log_1_gamma0[i] = np.nan_to_num(log_1_gamma0[i])
        log_gamma1[i] = np.nan_to_num(log_gamma1[i])
        log_1_gamma1[i] = np.nan_to_num(log_1_gamma1[i])

    return gamma0, gamma1, log_gamma0, log_gamma1, log_1_gamma0, log_1_gamma1
"""
def initiation(rowX):
    types = [0, 1]                                      #types of tweets
    input_ar = np.arange(D).flatten()                   #array to be used in finding initiators
    for i in range(N):                                  #for every tweet, we pick type of tweet and then the user who initiated it
        tt = np.random.choice(types, p=[pi[0],pi[1]])       #picking type of tweet with probability pi(k)
        tweet_types[i] = tt
        if tt == 0:
            xu = np.random.choice(input_ar, 1, p=phi0)[0]
        elif tt == 1:
            xu = np.random.choice(input_ar, 1, p=phi1)[0]
        rowX[i] = int(xu)                               #rowX stores the initiator of every tweet

    return tweet_types, rowX
"""
"""
def fwd_procedure(i):
    return check_neighbors(i)
"""
"""

def forwarding():
    global rowY
    global colY
    global W_list

    num_cores = cpu_count()
    pool = Pool(num_cores)
    results = pool.map_async(fwd_procedure, range(N))
    pool.close()
    pool.join()
    res = results.get()

    for item in res:
        rowY.append(item[0])
        colY.append(item[1])
        W_list.append(item[2])

    rowY = [item for sublist in list(rowY) for item in sublist]
    colY = [item for sublist in list(colY) for item in sublist]
    W_list = [item for sublist in list(W_list) for item in sublist]

    return
"""
"""
def check_neighbors(tweet):
    usr_init = rowX[tweet]                                      #the initiator of the tweet
    tw = int(tweet)                                             #tweet id
    nb_list = [(p, usr_init, tw) for p in nbrs[usr_init]]       #[(neighbor, initiator, tweet) for all neighbors of the initiator]

    wll = []
    rowyy = []
    colyy = []

    while nb_list:                                              #queue that stores triples (neighbor, user, tweet) to examine viewings and retweets
        entry = nb_list.pop(0)
        person = int(entry[0])
        u_init = int(entry[1])

        # we first check that the neighbor was not the initiator of the tweet
        # because if the user was one of the initiators then yun is always 0 and the user will not forward this tweet
        #if not the initiator, continue

        if rowX[tw] != person:
            wll.append([person, u_init, tw])                    #tweet is marked as seen from user u by user v
            not_retweeted_before = True

            if person in rowyy:
                indY = [int(i) for i, j in enumerate(rowyy) if int(j) == person]  #returns all indices in rowY where we find "person"
                for idx in indY:
                    if colyy[idx] == tw:                        #if retweeted before by this user, then ignore next retweets
                        not_retweeted_before = False
                        if [u_init, person, tw] in wll:
                            wll.remove([u_init, person, tw])
            if not_retweeted_before:
                if tweet_types[tw] == 0:                                #if the type of the tweet is 0, then
                    yu = np.random.binomial(1, p=gamma0[person])        #retweeted/not retweeted as type 0
                else:                                                   #if the type of the tweet is 1, then
                    yu = np.random.binomial(1, p=gamma1[person])        #retweeted/not retweeted as type 1
                if yu == 1:                                             #if retweeted, the forwarder is now the "initiator"
                    for nb in nbrs[person]:
                        nb_list.append((nb, person,tw))  #append the neighbors of the user who retweeted this tweet so that we check further forwarding of the tweet

                    rowyy.append(person)                 #mark tweet as forwarded by user
                    colyy.append(tw)
    return rowyy, colyy, wll
"""
if __name__ == "__main__":
    print ("DATASET CREATION")

    k = 2
    N = int(float(sys.argv[1]))     #number of tweets
    a = float(sys.argv[2])          #parameter a for gamma1
    b = float(sys.argv[3])          #parammeter b for gamma0
    choice = int(sys.argv[4])  #choice indicates whether to use existing graph or create new one
    # if choice = 0 then use existing graph, if choice =1 then create new random graph
    # with predefined number of nodes, density
    fname = str(sys.argv[5])        #csv filename of the existing follow graph
    loops_count = int(float(sys.argv[6]))   #how many iterations of the algorithm/the parameters generation
    option = str(sys.argv[7])   #option parameter for phi creation - m marginal, r uniformly at random, u uniform
    rep = str(sys.argv[8])      #number of repetitions(????)
    dns = float(sys.argv[9])    #graph density
    D = int(sys.argv[10])       #number of nodes = number of users in my graph
    t1 = time.time()
    #fname = "/Users/isidora/Desktop/Research/twitter_gen/follow_graphs/follow_graph_beefban.csv"
    fname ='./edgesMessi.csv'

    print ("loops_count")

    if choice == 0:     #creating the graph from existing edgelist
        print ("graph from dataset")
        names = ['user1','user2']
        users_content = users_to_df(fname, names)

        G = nx.from_pandas_edgelist(users_content,source=names[0],target=names[1])      #creating graph
        nbrs = [tuple(nx.all_neighbors(G, v))for v in G]                                #list of tuples neighbors of every user
        D=len(nbrs)
        print ("number of users = ", D)
        dns = nx.density(G)


    elif choice ==1:        #creating new random graph with given number of nodes and density
        #print ("random graph")
        #g_random = nx.barabasi_albert_graph(D,D/50)
        g_random = nx.gnp_random_graph(D, dns)                 #creating a random graph of the users with D vertices and probability for edge creation=1/D
        nbrs = [tuple(nx.all_neighbors(g_random, v))for v in g_random]  #list of tuples neighbors of every user
        #print ("number of users = ", len(nbrs))
        #print
        G = g_random


    dns2 = nx.density(G)
    print ("Graph Density", dns)
    print
    print ("Graph Density 2", dns2)
    print

    nc = nx.node_connectivity(G)
    print ("Node Connectivity = ", nc)
    print

    ec = nx.edge_connectivity(G)
    print ("Edge Connectivity = ", ec)
    print


    "SUBFOLDERS FOR THE CURRENT EXPERIMENT"
    "---------------------------------------------------------------------------------------------------------------"
    here = os.path.dirname(os.path.realpath(__file__))
    subdirout = 'output'
    filepath0 = os.path.join(here, subdirout)
    if not os.path.exists(filepath0):
        os.makedirs(filepath0)

    subdird = 'datasets'
    filepathd = os.path.join(here, subdird)
    if not os.path.exists(filepathd):
        os.makedirs(filepathd)
    dest = filepathd
    subdir = 'N' + str(N) + 'D'+str(D)+"d"+str(dns)+'a' + str(a) + 'b' + str(b)+'_'+rep
    dest = os.path.join(dest, subdir)
    if not os.path.exists(dest):
        os.makedirs(dest)

    """DEFINITIONS"""
    "---------------------------------------------------------------------------------------------------------------"

    rowX = np.zeros(N, dtype=int)               # rowX stores the initiators of the tweets - every tweet has only one initiator
    colX = np.arange(N)                         # colX stores all tweets' ids, [1...N]
    dataX = np.ones(N)                          # dataX: array of 1's of length N
    pi = np.zeros((loops_count,2), dtype=np.float128)

    rowY = []                                   # initializing rowY, colY to be used for forwarding
    colY = []
    W_list = []                                 # W_list: list of triplets (i,j,k): user i saw tweet k from user j

    tweet_types = np.zeros(N)
    phi0 = np.zeros(D, dtype=np.float128)
    phi1 = np.zeros(D, dtype=np.float128)
    log_phi0 = np.zeros(D, dtype=np.float128)
    log_phi1 = np.zeros(D, dtype=np.float128)

    gamma0 = np.zeros(D, dtype=np.float128)
    gamma1 = np.zeros(D, dtype=np.float128)
    log_gamma0 = np.zeros(D, dtype=np.float128)
    log_gamma1 = np.zeros(D, dtype=np.float128)
    log_1_gamma0 = np.zeros(D, dtype=np.float128)
    log_1_gamma1 = np.zeros(D, dtype=np.float128)

    """
    "ASSIGN PI"
    "---------------------------------------------------------------------------------------------------------------"

    pi= prop_pi()                               #probability that a tweet belongs to one of k=2 categories
    
    "ASSIGN PHIs"
    "---------------------------------------------------------------------------------------------------------------"

    phi0, phi1, log_phi0, log_phi1 = assign_user_phi(D, b, option)
    print ("assigned phis")
    #print

    "ASSIGN GAMMAs"
    "---------------------------------------------------------------------------------------------------------------"

    gamma0, gamma1, log_gamma0, log_gamma1, log_1_gamma0, log_1_gamma1 = assign_user_gamma(D, a)
    print ("assigned gammas")
    #print

    "INITIATION"
    "---------------------------------------------------------------------------------------------------------------"
    
    rowX =  hkl.load(source + '/rowX.hkl')  # initiators
    #tweet_types, rowX = initiation(rowX)
    print ("done with initiation")
    print rowX

    "FORWARDING"
    "---------------------------------------------------------------------------------------------------------------"

    forwarding()
    #print ("done with forwarding")
    #print


    "VIEWERS"
    "---------------------------------------------------------------------------------------------------------------"

    W_l = list(map(list, zip(*W_list)))         #list of triples containing (user1, user2, tweet)-user1 saw tweet from user2
    W = COO(W_l, 1, shape=(D, D, N))

    t2 = time.time()
    diff = t2 - t1
    #print ("time(sec) for init+fwd= ", diff)


    "WRITE TO FILES"
    "---------------------------------------------------------------------------------------------------------------"

    hkl.dump(pi, dest + "/pi_init.hkl", mode='w',dtype=np.float128)
    hkl.dump(tweet_types, dest + "/tweet_types.hkl", mode='w',dtype=np.float64)
    hkl.dump(log_phi0, dest + "/log_phi0_init.hkl", mode='w',dtype=np.float128)
    hkl.dump(log_phi1, dest + "/log_phi1_init.hkl", mode='w',dtype=np.float128)
    hkl.dump(log_gamma0, dest + "/log_gamma0_init.hkl", mode='w',dtype=np.float128)
    hkl.dump(log_gamma1, dest + "/log_gamma1_init.hkl", mode='w',dtype=np.float128)
    hkl.dump(log_1_gamma0, dest + "/log_1_gamma0_init.hkl", mode='w',dtype=np.float128)
    hkl.dump(log_1_gamma1, dest + "/log_1_gamma1_init.hkl", mode='w',dtype=np.float128)

    hkl.dump(rowX, dest + "/rowX.hkl", mode='w')
    hkl.dump(rowY, dest + "/rowY.hkl", mode='w')
    hkl.dump(colY, dest + "/colY.hkl", mode='w')
    hkl.dump(nbrs, dest + "/nbrs.hkl", mode='w')

    save_npz(dest + '/W.npz', W)

    t3 = time.time()
    diff = t3 - t1
    #print ("time(sec) = ", diff)
    #print
    """



