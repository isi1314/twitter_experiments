from importing import *
from sparse import *
import random
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def viewers(i):
    wl = []
    usr_rt = int(rowY[i])
    tw = int(colY[i])
    usr_init = int(rowX[tw])
    for nb in nbrs[usr_rt]:
        if nb != rowX[tw]:
            wl.append([nb, usr_rt, tw])
    return wl


def create_W_fwd():
    Wl = []
    num_cores = cpu_count()

    # adding the neighbors of the retweeters and the neighbors of the initiators as viewers
    pool = Pool(num_cores)
    print ("number of cores = ", num_cores)
    num_retweets = len(rowY)
    results = pool.map_async(viewers, range(num_retweets))
    pool.close()
    pool.join()
    res = results.get()
    for item in res:
        Wl.append(item)
    Wl = [item for sublist in list(Wl) for item in sublist]

    return Wl


def initiators(i):
    wl = []
    usr_init = int(rowX[i])
    tw = int(i)
    for nb in nbrs[usr_init]:
        wl.append([nb, usr_init, tw])

    return wl


def create_W_init():
    Wll = []
    num_cores = cpu_count()
    # adding the neighbors of the retweeters and the neighbors of the initiators as viewers
    pool = Pool(num_cores)
    print ("number of cores = ", num_cores)
    results = pool.map_async(initiators, range(N))
    pool.close()
    pool.join()
    res = results.get()
    for item in res:
        Wll.append(item)
    Wll = [item for sublist in list(Wll) for item in sublist]

    return Wll


def assign_random_parameter_values():  # assigning random initial values to start running EM

    init0 = np.random.uniform(0, 1, D)
    init1 = np.random.uniform(0, 1, D)
    init0 = init0 / init0.sum()
    init1 = init1 / init1.sum()

    init0 = np.log(init0, dtype=np.float128)
    init1 = np.log(init1, dtype=np.float128)

    g0 = np.random.uniform(0, 1, D)
    g1 = np.random.uniform(0, 1, D)
    p0 = np.random.uniform(0, 1, 1)[0]
    #p0 = 0.5
    p1 = 1 - p0

    """
    print
    print ("initial p0, p1= ", p0, p1)
    print
    """

    tmp1 = np.ones(shape=g0.shape) - g0
    tmp2 = np.ones(shape=g1.shape) - g1

    """
    print
    print ("initial g0 = ", g0)
    print ("initial g1 = ", g1)
    print
    """

    log_g0 = np.log(g0, dtype=np.float128)
    log_g1 = np.log(g1, dtype=np.float128)
    log_1_g0 = np.log(tmp1, dtype=np.float128)
    log_1_g1 = np.log(tmp2, dtype=np.float128)
    logp0 = np.log(p0, dtype=np.float128)
    logp1 = np.log(p1, dtype=np.float128)

    return init0, init1, log_g0, log_g1, log_1_g0, log_1_g1, logp0, logp1


def assign_known_parameter_values(source):
    # in position 0 we store the originally created and correct values of the parameters

    pinit = hkl.load(source + "/pi_init.hkl")
    logp0 = np.log(pinit[0])
    logp1 = np.log(pinit[1])

    init0 = hkl.load(source + '/log_phi0_init.hkl')
    init1 = hkl.load(source + '/log_phi1_init.hkl')
    g0 = hkl.load(source + '/log_gamma0_init.hkl')
    g1 = hkl.load(source + '/log_gamma1_init.hkl')
    g00 = hkl.load(source + '/log_1_gamma0_init.hkl')
    g11 = hkl.load(source + '/log_1_gamma1_init.hkl')


    return init0, init1, g0, g1, g00, g11, logp0, logp1


def assign_fixed_parameter_values():  # assigning initial values to start running EM
    global W

    init0 = np.zeros(D)
    init1 = np.zeros(D)
    fwd0 = np.zeros(D)
    fwd1 = np.zeros(D)
    seen0 = np.zeros(D)
    seen1 = np.zeros(D)


    for idx, u in np.ndenumerate(rowX):
        if tweet_types[idx] == 0:
            init0[u] += 1  # number of tweets of type 0 this user initiated
        else:
            init1[u] += 1  # number of tweets of type 1 this user initiated

    for item in zip(W.coords[0], W.coords[1], W.coords[2]):

        if tweet_types[item[2]] == 0:
            seen0[item[0]] += 1  # number of tweets of type 0 this user saw
            if Y[item[0], item[2]] == 1:
                fwd0[item[0]] += 1  # number of tweets of type 0 this user saw and forwarded
        else:
            seen1[item[0]] += 1  # number of tweets of type 1 this user saw
            if Y[item[0], item[2]] == 1:
                fwd1[item[0]] += 1  # number of tweets of type 1 this user saw and forwarded

    in0 = np.divide(init0, count0, dtype=np.float128)
    in1 = np.divide(init1, count1, dtype=np.float128)

    in0 = np.nan_to_num(in0)
    in1 = np.nan_to_num(in1)

    g0 = np.divide(fwd0, seen0, dtype=np.float128)
    g1 = np.divide(fwd1, seen1, dtype=np.float128)

    g0 = np.nan_to_num(g0)
    g1 = np.nan_to_num(g1)

    p0 = count0 / N
    p1 = count1 / N

    g00 = np.log(1 - g0, dtype=np.float128)
    g0 = np.log(g0, dtype=np.float128)
    g11 = np.log(1 - g1, dtype=np.float128)
    g1 = np.log(g1, dtype=np.float128)

    g00 = np.nan_to_num(g00)
    g11 = np.nan_to_num(g11)

    lp0 = np.log(p0, dtype=np.float128)
    lp1 = np.log(p1, dtype=np.float128)

    return in0, in1, g0, g1, g00, g11, lp0, lp1


def f_update(loop_num):
    global f0
    global f1
    global init_ll
    global fwd_ll
    global ll
    global log_f0
    global log_f1

    lpi0 = logpi[loop_num][0]  # log pi[0] for this iteration
    lpi1 = logpi[loop_num][1]  # log pi[1] for this iteration

    sumX0 = log_phi0[loop_num, rowX]  # sum(xun log(phi0)) over all users
    sumX1 = log_phi1[loop_num, rowX]  # sum(xun log(phi1)) over all users

    sum_Y0 = Y_csrt.multiply(log_gamma0[loop_num]) + Y1_csrt.multiply(log_1_gamma0[loop_num])  # yun*log(gamma0)+(1-yun)*log(1-gamma0)
    sum_Y1 = Y_csrt.multiply(log_gamma1[loop_num]) + Y1_csrt.multiply(log_1_gamma1[loop_num])  # yun*log(gamma1)+(1-yun)*log(1-gamma1)

    sum_Y0.data = np.nan_to_num(sum_Y0.data)
    sum_Y1.data = np.nan_to_num(sum_Y1.data)

    sumY0 = np.sum(sum_Y0.transpose().multiply(ss), axis=0, dtype=np.float128)  # sum(wuvn(yun*log(gamma0)+(1-yun)*log(1-gamma0)))
    sumY1 = np.sum(sum_Y1.transpose().multiply(ss), axis=0, dtype=np.float128)

    resY0 = np.asarray(sumX0 + sumY0 + lpi0, dtype=np.float128)[0]  # =log(numerator0) =log(A0)
    resY1 = np.asarray(sumX1 + sumY1 + lpi1, dtype=np.float128)[0]  # =log(numerator1) =log(A1)

    sum_log = np.logaddexp(resY0, resY1, dtype=np.float128)  # =log(numerator0+numerator1) = log(exp(log(numerator0))+ exp(log(numerator1)))
    sum_log = np.nan_to_num(sum_log)

    log_f0[loop_num] = np.subtract(resY0, sum_log, dtype=np.float128)  # log(f(zn0)) = log(A0)- log(A0+A1)
    log_f1[loop_num] = np.subtract(resY1, sum_log, dtype=np.float128)  # log(f(zn1)) = log(A1)- log(A0+A1)

    f0[loop_num] = np.exp(log_f0[loop_num], dtype=np.float128)
    f1[loop_num] = np.exp(log_f1[loop_num], dtype=np.float128)

    f0[loop_num] = np.nan_to_num(f0[loop_num])
    f1[loop_num] = np.nan_to_num(f1[loop_num])

    init0 = np.add(lpi0, sumX0, dtype=np.float128)  # initiation log-likelihood of type0
    init1 = np.add(lpi1, sumX1, dtype=np.float128)  # initiation log-likelihood of type1
    init = np.logaddexp(init0, init1, dtype=np.float128)  # initiation log-likelihood of both types for one tweet
    init = np.nan_to_num(init)

    init_ll[loop_num] = np.sum(init, dtype=np.float128)

    fwd0 = np.add(lpi0, sumY0, dtype=np.float128)  # forwarding log-likelihood of type0
    fwd1 = np.add(lpi1, sumY1, dtype=np.float128)  # forwarding log-likelihood of type1
    fwd = np.logaddexp(fwd0, fwd1, dtype=np.float128)  # forwarding log-likelihood
    fwd = np.nan_to_num(fwd)

    fwd_ll[loop_num] = np.sum(fwd, dtype=np.float128)

    ll0 = np.add(fwd0, sumX0, dtype=np.float128)
    ll1 = np.add(fwd1, sumX1, dtype=np.float128)
    ll_array = np.logaddexp(ll0, ll1, dtype=np.float128)
    ll_array = np.nan_to_num(ll_array)
    ll[loop_num] = np.sum(ll_array, dtype=np.float128)  # log-likelihood

    return


def pi_update(loop_num):
    global pi
    global f0
    global f1
    global logpi

    pi0 = np.sum(f0[loop_num], dtype=np.float128) / N
    pi1 = np.sum(f1[loop_num], dtype=np.float128) / N

    logpi[loop_num + 1][0] = np.log(pi0, dtype=np.float128)
    logpi[loop_num + 1][1] = np.log(pi1, dtype=np.float128)

    logpi[loop_num + 1][0] = np.nan_to_num(logpi[loop_num + 1][0])
    logpi[loop_num + 1][1] = np.nan_to_num(logpi[loop_num + 1][1])

    return

def phi_update(loop_num):
    global log_phi0
    global log_phi1
    global Xs
    global f0t
    global f1t

    nom0 = Xs * f0t
    nom1 = Xs * f1t

    den0 = np.sum(nom0, dtype=np.float128)  # sum of numerator0 for all users
    den1 = np.sum(nom1, dtype=np.float128)  # sum of numerator1 for all users

    n0 = np.divide(nom0, den0, dtype=np.float128)
    n1 = np.divide(nom1, den1, dtype=np.float128)

    n0 = np.nan_to_num(n0)
    n1 = np.nan_to_num(n1)

    log_phi0[loop_num + 1] = np.log(n0, dtype=np.float128)
    log_phi1[loop_num + 1] = np.log(n1, dtype=np.float128)

    log_phi0[loop_num + 1] = np.nan_to_num(log_phi0[loop_num + 1])
    log_phi1[loop_num + 1] = np.nan_to_num(log_phi1[loop_num + 1])

    return


def gamma_update(loop_num):
    global log_gamma0
    global log_gamma1
    global log_1_gamma0
    global log_1_gamma1
    global f0
    global f1
    global Y

    n0 = ss_csr.multiply(f0[loop_num])
    den0 = np.asarray(np.sum(n0, axis=1, dtype=np.float128), dtype=np.float128)
    nom0 = np.asarray(np.sum(n0.multiply(Y), axis=1, dtype=np.float128), dtype=np.float128)

    n1 = ss_csr.multiply(f1[loop_num])
    den1 = np.asarray(np.sum(n1, axis=1, dtype=np.float128), dtype=np.float128)
    nom1 = np.asarray(np.sum(n1.multiply(Y), axis=1, dtype=np.float128), dtype=np.float128)

    r0 = np.divide(nom0, den0, dtype=np.float128)
    r1 = np.divide(nom1, den1, dtype=np.float128)

    r0 = np.nan_to_num(r0)
    r1 = np.nan_to_num(r1)
    tmp1 = np.ones(shape=r0.shape) - r0
    tmp2 = np.ones(shape=r1.shape) - r1

    r0.shape = (D)
    r1.shape = (D)
    tmp1.shape = (D)
    tmp2.shape = (D)
    log_gamma0[loop_num + 1] = np.log(r0, dtype=np.float128)
    log_gamma1[loop_num + 1] = np.log(r1, dtype=np.float128)
    log_1_gamma0[loop_num + 1] = np.log(tmp1, dtype=np.float128)
    log_1_gamma1[loop_num + 1] = np.log(tmp2, dtype=np.float128)

    log_gamma0[loop_num + 1] = np.nan_to_num(log_gamma0[loop_num + 1])
    log_gamma1[loop_num + 1] = np.nan_to_num(log_gamma1[loop_num + 1])
    log_1_gamma0[loop_num + 1] = np.nan_to_num(log_1_gamma0[loop_num + 1])
    log_1_gamma1[loop_num + 1] = np.nan_to_num(log_1_gamma1[loop_num + 1])

    return


if __name__ == "__main__":
    print ("EM ALGORITHM")
    N = int(sys.argv[1])                #number of tweets
    a = float(sys.argv[2])              #parameter a for gamma1
    b = float(sys.argv[3])              #parameter b for phi1
    loops_count = int(sys.argv[4])      #number of times I will be running the algorithm
    rep = str(sys.argv[5])              #number of iteration
    dns = float(sys.argv[6])            #density of the users follow graph
    vnum = int(sys.argv[7])             #how many times I will be running EM with the same initial data
    t1 = time.time()    
    D = int(sys.argv[8])                #number of users

    print "done with reading args"
    print

    """
    "SUBFOLDERS FOR THE CURRENT EXPERIMENT"
    "---------------------------------------------------------------------------------------------------"

    here = os.path.dirname(os.path.realpath(__file__))

    filepath0 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    paths = 'N' + str(N) + 'D' + str(D) + "d" + str(dns) + 'a' + str(a) + 'b' + str(b) + '_' + rep
    pathd = paths
    source = os.path.join(filepath0, paths)
    dest = os.path.join(filepath, pathd)
    if not os.path.exists(dest):
        os.makedirs(dest)

    nbrs = hkl.load(source + '/nbrs.hkl')       #loading the neighbours
    D = len(nbrs)                               #length of the neighbors list = number of users
    # print ("number of users = ", len(nbrs))
    """

    "DEFINITIONS"
    "-----------------------------------------------------------------------------------------------------"
    # initializing with very small numbers instead of 0

    pi = np.zeros((loops_count, 2), dtype=np.float128)
    logpi = np.zeros((loops_count, 2), dtype=np.float128)
    f0 = np.zeros((loops_count - 1, N), dtype=np.float128)
    f1 = np.zeros((loops_count - 1, N), dtype=np.float128)

    log_phi0 = np.zeros((loops_count, D), dtype=np.float128)
    log_phi1 = np.zeros((loops_count, D), dtype=np.float128)
    log_gamma0 = np.zeros((loops_count, D), dtype=np.float128)
    log_gamma1 = np.zeros((loops_count, D), dtype=np.float128)
    log_1_gamma0 = np.zeros((loops_count, D), dtype=np.float128)
    log_1_gamma1 = np.zeros((loops_count, D), dtype=np.float128)

    log_pi_sum = np.zeros((loops_count, 2), dtype=np.float128)
    log_phi0_sum = np.zeros((loops_count, D), dtype=np.float128)
    log_phi1_sum = np.zeros((loops_count, D), dtype=np.float128)
    log_gamma0_sum = np.zeros((loops_count, D), dtype=np.float128)
    log_gamma1_sum = np.zeros((loops_count, D), dtype=np.float128)
    log_1_gamma0_sum = np.zeros((loops_count, D), dtype=np.float128)
    log_1_gamma1_sum = np.zeros((loops_count, D), dtype=np.float128)

    f0_sum = np.zeros((loops_count - 1, N), dtype=np.float128)
    f1_sum = np.zeros((loops_count - 1, N), dtype=np.float128)
    log_f0_sum = np.zeros((loops_count - 1, N), dtype=np.float128)
    log_f1_sum = np.zeros((loops_count - 1, N), dtype=np.float128)

    log_pi_avg = np.zeros((loops_count, 2), dtype=np.float128)
    log_phi0_avg = np.zeros((loops_count, D), dtype=np.float128)
    log_phi1_avg = np.zeros((loops_count, D), dtype=np.float128)
    log_gamma0_avg = np.zeros((loops_count, D), dtype=np.float128)
    log_gamma1_avg = np.zeros((loops_count, D), dtype=np.float128)
    log_1_gamma0_avg = np.zeros((loops_count, D), dtype=np.float128)
    log_1_gamma1_avg = np.zeros((loops_count, D), dtype=np.float128)

    f0_avg = np.zeros((loops_count - 1, N), dtype=np.float128)
    f1_avg = np.zeros((loops_count - 1, N), dtype=np.float128)
    log_f0_avg = np.zeros((loops_count - 1, N), dtype=np.float128)
    log_f1_avg = np.zeros((loops_count - 1, N), dtype=np.float128)

    log_f0 = np.zeros((loops_count - 1, N), dtype=np.float128)
    log_f1 = np.zeros((loops_count - 1, N), dtype=np.float128)

    init_ll = np.zeros(loops_count - 1, dtype=np.float128)
    fwd_ll = np.zeros(loops_count - 1, dtype=np.float128)
    ll = np.zeros(loops_count - 1, dtype=np.float128)

    print "done with definitions"
    print

    """INITIATION MATRICES"""
    "-----------------------------------------------------------------------------------------------------"

    rowX = hkl.load(source + '/rowX.hkl')  # initiators
    rowX = rowX.astype(int)
    colX = np.arange(N)  # colX stores all tweets' ids, [1...N]
    dataX = np.ones(N)  # dataX: array of 1's of length N
    Xs = csr_matrix((dataX, (rowX, colX)), shape=(D, N), dtype=np.float128)
    # print ("Density of X matrix = ", Xs.getnnz() / float(D * N))

    """FORWARDING MATRICES"""
    "-----------------------------------------------------------------------------------------------------"

    rowY = hkl.load(source + '/rowY.hkl')
    colY = hkl.load(source + '/colY.hkl')
    dataY = np.ones(len(rowY))
    # Y = rand(D, N, density=0.8, format='csr')          #used when testing for random matrix
    Y = csr_matrix((dataY, (rowY, colY)), shape=(D, N))
    Y1 = np.ones((D, N)) - Y  # Y1 = 1-Y
    Y_csrt = Y.transpose()
    Y1_csrt = csr_matrix(Y1.transpose())

    Y_counts =Y.sum(axis=0)
    Ytest = np.asarray(Y_counts).reshape(-1)
    Y_counts1 =Y.sum(axis=1)

    Y_counts1=np.asarray(Y_counts1)


    "VIEWINGS' MATRICES"
    "-----------------------------------------------------------------------------------------------------"

    # W_list = create_W_fwd()
    # w_inits= create_W_init()
    # W_list = W_list + w_inits
    # W_l = list(map(list, zip(*W_list)))  # list of tuples containing (user, user, tweet)
    # W_list = sorted(W_list)
    # dedupW = [W_list[i] for i in range(len(W_list)) if i == 0 or W_list[i] != W_list[i - 1]]
    # print (len(dedupW))

    # if sorted(W_list)==sorted(W_list_f):
    # print ("success")

    # W = COO(W_l, 1, shape=(D, D, N))
    # OR
    # W = random((D,D,N), density=0.015)
    # OR

    # for synthetic data, we load the matrix we have from previously creating it
    W = load_npz(source + '/' + 'W.npz')

    # print ("Density of W matrix = ", W.density)
    sumW = W.sum(axis=1)  # summing over axis 1 to get the number of times every user saw each tweet
    # print ("Density of sumW matrix = ", sumW.density)


    ss = sumW.to_scipy_sparse()
    ss_csr = ss.tocsr()
    # print ("Density of ss matrix = ", ss.getnnz() / float(D * N))

    tweet_types = hkl.load(source + '/tweet_types.hkl')
    count1 = tweet_types.sum()  # number of tweets of type 1
    count0 = N - count1  # number of tweets of type 0
    print ("count0 = ", count0)
    print ("count1 = ", count1)


    "ASSIGNING INITIAL VALUES TO PARAMETERS"
    "-----------------------------------------------------------------------------------------------------"
    # either random or fixed based on data statistics

    log_phi0[0], log_phi1[0],log_gamma0[0], log_gamma1[0], log_1_gamma0[0], log_1_gamma1[0], logpi[0][0], logpi[0][1] = assign_random_parameter_values()
    #log_phi0[0], log_phi1[0],log_gamma0[0], log_gamma1[0], log_1_gamma0[0], log_1_gamma1[0], logpi[0][0], logpi[0][1] = assign_fixed_parameter_values()
    #log_phi0[0], log_phi1[0],log_gamma0[0], log_gamma1[0], log_1_gamma0[0], log_1_gamma1[0], logpi[0][0], logpi[0][1] = assign_known_parameter_values(source)

    """
    log_gamma0[0] = hkl.load(source + '/log_gamma0_in.hkl')
    log_gamma1[0] = hkl.load(source + '/log_gamma1_in.hkl')
    log_1_gamma0[0] = hkl.load(source + '/log_1_gamma0_in.hkl')
    log_1_gamma1[0] = hkl.load(source + '/log_1_gamma1_in.hkl')
    #logpi[0] = hkl.load(source + '/log_pi0_in.hkl')

    hkl.dump(logpi[0], source + "/log_pi0_in.hkl", mode='w')
    hkl.dump(log_gamma0[0], source + "/log_gamma0_in.hkl", mode='w')
    hkl.dump(log_gamma1[0], source + "/log_gamma1_in.hkl", mode='w')
    hkl.dump(log_1_gamma0[0], source + "/log_1_gamma0_in.hkl", mode='w')
    hkl.dump(log_1_gamma1[0], source + "/log_1_gamma1_in.hkl", mode='w')
    """

    "EM"
    "-----------------------------------------------------------------------------------------------------"

    for i in range(vnum):  # running EM algorithm 10 times with the same initializations
        dest_i = dest + '/' + str(i) + '/'

        for count in range(loops_count - 1):  # algorithm iterations
            f_update(count)
            f0t = np.asarray(f0[count].T, dtype=np.float128)
            f1t = np.asarray(f1[count].T, dtype=np.float128)
            pi_update(count)
            phi_update(count)
            gamma_update(count)

        log_pi_sum = np.add(log_pi_sum, logpi, dtype=np.float128)
        log_phi0_sum = np.add(log_phi0_sum, log_phi0, dtype=np.float128)
        log_phi1_sum = np.add(log_phi1_sum, log_phi1, dtype=np.float128)
        log_gamma0_sum = np.add(log_gamma0_sum, log_gamma0, dtype=np.float128)
        log_gamma1_sum = np.add(log_gamma1_sum, log_gamma1, dtype=np.float128)
        log_1_gamma0_sum = np.add(log_1_gamma0_sum, log_1_gamma0, dtype=np.float128)
        log_1_gamma1_sum = np.add(log_1_gamma1_sum, log_1_gamma1, dtype=np.float128)
        log_f0_sum = np.add(log_f0_sum, log_f0, dtype=np.float128)
        log_f1_sum = np.add(log_f1_sum, log_f1, dtype=np.float128)
        f0_sum = np.add(f0_sum, f0, dtype=np.float128)
        f1_sum = np.add(f1_sum, f1, dtype=np.float128)



    # CALCULATION OF AVERAGE OVER 10 ITERATIONS
    "-----------------------------------------------------------------------------------------------------"
    log_pi_avg = np.divide(log_pi_sum, vnum)
    log_phi0_avg = np.divide(log_phi0_sum, vnum)
    log_phi1_avg = np.divide(log_phi1_sum, vnum)
    log_gamma0_avg = np.divide(log_gamma0_sum, vnum)
    log_gamma1_avg = np.divide(log_gamma1_sum, vnum)
    log_1_gamma0_avg = np.divide(log_1_gamma0_sum, vnum)
    log_1_gamma1_avg = np.divide(log_1_gamma1_sum, vnum)
    log_f0_avg = np.divide(log_f0_sum, vnum)
    log_f1_avg = np.divide(log_f1_sum, vnum)
    f0_avg = np.divide(f0_sum, vnum)
    f1_avg = np.divide(f1_sum, vnum)




    for i in range(loops_count - 1):
        lpi0 = log_pi_avg[i][0]  # log pi[0] for this iteration
        lpi1 = log_pi_avg[i][1]  # log pi[1] for this iteration

        sumX0 = log_phi0_avg[i, rowX]  # sum(xun log(phi0)) over all users
        sumX1 = log_phi1_avg[i, rowX]  # sum(xun log(phi1)) over all users

        init0 = np.add(lpi0, sumX0, dtype=np.float128)  # initiation log-likelihood of type0
        init1 = np.add(lpi1, sumX1, dtype=np.float128)  # initiation log-likelihood of type1
        init = np.logaddexp(init0, init1, dtype=np.float128)  # initiation log-likelihood of both types for one tweet
        init = np.nan_to_num(init)

        init_ll[i] = np.sum(init, dtype=np.float128)

        sum_Y0 = Y_csrt.multiply(log_gamma0_avg[i]) + Y1_csrt.multiply(log_1_gamma0_avg[i])  # yun*log(gamma0)+(1-yun)*log(1-gamma0)
        sum_Y1 = Y_csrt.multiply(log_gamma1_avg[i]) + Y1_csrt.multiply(log_1_gamma1_avg[i])  # yun*log(gamma1)+(1-yun)*log(1-gamma1)
        sum_Y0.data = np.nan_to_num(sum_Y0.data)
        sum_Y1.data = np.nan_to_num(sum_Y1.data)

        sumY0 = np.sum(sum_Y0.transpose().multiply(ss), axis=0,dtype=np.float128)  # sum(wuvn(yun*log(gamma0)+(1-yun)*log(1-gamma0)))
        sumY1 = np.sum(sum_Y1.transpose().multiply(ss), axis=0, dtype=np.float128)

        fwd0 = np.add(lpi0, sumY0, dtype=np.float128)  # forwarding log-likelihood of type0
        fwd1 = np.add(lpi1, sumY1, dtype=np.float128)  # forwarding log-likelihood of type1
        fwd = np.logaddexp(fwd0, fwd1, dtype=np.float128)  # forwarding log-likelihood
        fwd = np.nan_to_num(fwd)

        fwd_ll[i] = np.sum(fwd, dtype=np.float128)

        ll0 = np.add(fwd0,sumX0,dtype=np.float128 )
        ll1 =  np.add(fwd1,sumX1,dtype=np.float128 )
        ll_array = np.logaddexp(ll0,ll1, dtype=np.float128)
        ll_array = np.nan_to_num(ll_array)
        ll[i] = np.sum(ll_array, dtype=np.float128)  # log-likelihood



    "WRITE TO FILES"
    "-----------------------------------------------------------------------------------------------------"
    dest_i = dest + '/'
    if not os.path.exists(dest_i):
        os.makedirs(dest_i)
    hkl.dump(f0_avg, dest_i + "f0.hkl", mode='w')
    hkl.dump(f1_avg, dest_i + "f1.hkl", mode='w')
    hkl.dump(log_pi_avg, dest_i + "log_pi.hkl", mode='w')
    hkl.dump(log_f0_avg, dest_i + "log_f0.hkl", mode='w')
    hkl.dump(log_f1_avg, dest_i + "log_f1.hkl", mode='w')
    hkl.dump(fwd_ll, dest_i + "fwd_ll.hkl", mode='w')
    hkl.dump(ll, dest_i + "ll.hkl", mode='w')
    hkl.dump(init_ll, dest_i + "init_ll.hkl", mode='w')
    hkl.dump(log_phi0_avg, dest_i + "log_phi0.hkl", mode='w')
    hkl.dump(log_phi1_avg, dest_i + "log_phi1.hkl", mode='w')
    hkl.dump(log_gamma0_avg, dest_i + "log_gamma0.hkl", mode='w')
    hkl.dump(log_gamma1_avg, dest_i + "log_gamma1.hkl", mode='w')
    hkl.dump(tweet_types, dest_i + "tweet_types.hkl", mode='w')

    """
    np.savetxt(dest_i + 'log_pi.txt', log_pi_avg, header='log_pi')
    np.savetxt(dest_i + 'log_gamma0.txt', log_gamma0_avg, header='log_gamma0')
    np.savetxt(dest_i + 'log_gamma1.txt', log_gamma1_avg, header='log_gamma1')
    np.savetxt(dest_i + 'log_1_gamma0.txt', log_1_gamma0_avg, header='log_1_gamma0')
    np.savetxt(dest_i + 'log_1_gamma1.txt', log_1_gamma1_avg, header='log_1_gamma1')
    np.savetxt(dest_i + 'fwd_ll.txt', fwd_ll, header='fwd_ll')
    np.savetxt(dest_i + 'll.txt', ll, header='ll')
    np.savetxt(dest_i + 'log_f0.txt', log_f0_avg, header='log_f0')
    np.savetxt(dest_i + 'log_f1.txt', log_f1_avg, header='log_f1')
    """

