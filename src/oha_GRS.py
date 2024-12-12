
import csv
from xmlrpc.client import Boolean
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def data_load_100k(file_name: str, verbose: bool): # file_name : u1, u2, u3, u4, u5
    """_summary_
    
    함수의 입력은 str 타입으로 u1, u2, u3, u4, u5만 받을 수 있다.
    이 이름이 의미하는 것은 ml-100k 폴더에 있는 u1.base, u2.base, u3.base, u4.base, u5.base 파일들이다.
    해당 파일들은 모두 ml-100k의 u.data에서 5-fold validation을 위해 split된 데이터이고, 이대로 사용하면 된다.
    user와 item의 번호는 0부터 시작하는 것으로 한다.

    Returns:
        전체 train set 행렬, train set 을 행렬로 변환시킨 train_set 과 test_set

    """

    if file_name == 'u1':
        file_path_train = './ml-100k/u1.base'
        file_path_test = './ml-100k/u1.test'

    elif file_name == 'u2':
        file_path_train = './ml-100k/u2.base'
        file_path_test = './ml-100k/u2.test'

    elif file_name == 'u3':
        file_path_train = './ml-100k/u3.base'
        file_path_test = './ml-100k/u3.test'

    elif file_name == 'u4':
        file_path_train = './ml-100k/u4.base'
        file_path_test = './ml-100k/u4.test'

    elif file_name == 'u5':
        file_path_train = './ml-100k/u5.base'
        file_path_test = './ml-100k/u5.test'

    else:
        print('data_load fail !!!!')
        return
    
    user_count = 943
    item_count = 1682


    # TOTAL_data :: user-item matrix
    df = pd.read_csv('./ml-100k/u.data', sep = '\s', names = ['userId', 'itemId', 'Rating', "timestamp"], header=None, na_filter=False)
    df = df[['userId','itemId','Rating']]

    TOTAL_data = df.pivot_table('Rating', index = 'userId',columns = 'itemId')

    # train :: user-item matrix
    df = pd.read_csv(file_path_train, sep = ',', names = ['userId', 'itemId', 'Rating'])
    R = df.pivot_table('Rating', index = 'userId',columns = 'itemId')


    # test :: user-item matrix
    df = pd.read_csv(file_path_test, sep = ',', names = ['userId', 'itemId', 'Rating'])
    T = df.pivot_table('Rating', index = 'userId',columns = 'itemId')


    if verbose == True:
        print(file_name, 'data_load success!')
        print('user_count: %d, item_count: %d'%(user_count, item_count))
        
    return TOTAL_data, R, T


def clustering(num, TOTAL_data, train_data, test_data):
    """_summary_

    Args:
        num (int): 그룹 개수
        TOTAL_data (pandas): 모든 데이터
        train_data (pandas): 훈련 데이터
        test_data (pandas): 학습 데이터

    Returns:
        pandas: 마지막 열에 0~(num-1)의 수를 가진 'cluster' 가 추가
    """
    # 1. clusturing
    km = KMeans(n_clusters=num, init='k-means++')
    cluster = km.fit(TOTAL_data)
    cluster_id = pd.DataFrame(cluster.labels_)                   # 모든  user에 클러스터 n 이 표시됨

    cluster_id.index = TOTAL_data.index                        # userId 칼럼을 인덱스로 설정 1~943
    cluster_id.rename(columns = {0 : 'cluster'}, inplace = True) # 모든  user의 클러스터


    ## 2. train, test 에 cluster 정보 추가
    train_data = pd.concat([train_data, cluster_id], axis=1, join='inner')
    test_data = pd.concat([test_data, cluster_id], axis=1, join='inner')

    return train_data, test_data


def Group_predict(train_data, nantoZero=True, tool='avg'):
    """_summary_

    Args:
        train_data (pandas): 마지막 열이 cluster 인 user-item matrix
        nantoZero(bool): nan 값을 0으로 바꿀것인지?
        tool(str): avg, lm (도출 방식)
    Returns:
        pandas: 클러스터 개수 만큼의 row 에 각 item 그룹별 평균 값 저장
    """

    #평균값을 담을 data frame
    train_data = train_data.set_index('cluster')

    if tool=='avg':
        train_data = train_data.groupby('cluster').mean()
    else:
        train_data = train_data.groupby('cluster').min()
    
    if nantoZero:
        train_data = train_data.fillna(0)

    return train_data


# maniac 행렬 도출
def Maniac(train_data, cluster_num):
    """_summary_

    Args:
        train_data (pandas)
        cluster_num (int)
    Returns:
        pandas: numn 그룹의 maniac 행렬
    """    
    #maniac = train_data[train_data.cluster == cluster_num].iloc[:,:-1]
    
    result = pd.DataFrame()
    for i in list(train_data[train_data.cluster==cluster_num].index):
        tmp = pd.DataFrame((train_data[train_data.cluster==cluster_num].mean()/train_data[train_data.cluster==cluster_num][train_data[train_data.cluster==cluster_num].index != i].mean())) # 해당 유저가 빠졌을 떄의 평균
        result = result.append(tmp.T.iloc[:,:-1].rename(index={0:i}))
    return (result-1).dropna(axis=1)


def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))


def Nash_Group_predict(train_data):
    rating = pd.DataFrame()

    # 각 클러스터별 회원 수 
    length = list(train_data.cluster.value_counts().sort_index())

    #클러스터 별로
    for i in range(len(length)):

        #클러스터 별 매니악 행렬 구함
        maniacs = Maniac(train_data, i)

        #클러스터의 평균을 구함  # 안 구해지는 아이템은 제거
        mean = train_data[train_data.cluster == i].iloc[:,:-1].mean(axis =0).dropna(inplace=False) # centroid
        mean.name = 'cl_mean'

        # 각 이용자 만큼 계산
        for j in tqdm(range(length[i])):
            user = train_data[train_data.cluster == i].iloc[j].dropna(inplace=False)
            user.name = 'user'

            maniac = maniacs.iloc[j]
            maniac.name = 'maniac'


            # # 평균 예측평점과 각 사용자 마다의 유사도 및 선호도 측정
            mean_user = pd.merge(mean, user, how='inner', left_index = True, right_index = True)

            # # 실제로 적용할 계산
            mean_user_maniac = pd.merge(mean, maniac, how='inner', left_index = True, right_index = True)

            if len(mean_user)==0: 
                continue

            #아이템이 하나만 겹치는 정도는 유사도를 측정할 수 없다 : 해당 유저가 대표와 얼마나 유사한 지 알 수 없기 때문이다.
            if len(mean_user) ==1:
                continue

            cos = cos_sim(mean_user.cl_mean.to_numpy(), mean_user.user.to_numpy())

            # 평균에 해당 평점예측에 따른 사용자의 선호도, 비선호도를 각각 계산했다.
            # 대표자의 예측 평점과 특정 사용자 간의 유사도 가중 평균을 도출한다.
            modified = (mean_user_maniac.cl_mean)   + (mean_user_maniac.maniac *(cos)) #(( mean_user.user - mean_user.cl_mean ) / 4  * cos)
            mean_modi = pd.concat([mean,modified], axis=1)
            mean = pd.DataFrame({'cl_mean':np.where(pd.notnull(mean_modi['cl_mean']) == True, mean_modi['cl_mean'], mean_modi[0])}, index=mean_modi.index)
        
        mean.columns = [str(i)]
        rating = rating.append(mean.T)

    return rating.fillna(0)



def miss_matching(train_data, test_data):
    """_summary_
    학습된 데이터랑 예측할 데이터의 열 데이터를 맞춰주기 위한 함수

    Args:
        train_data (pandas)
        test_data (pandas)

    Returns:
        train_data
        test_data
    """
    
    for c in train_data.columns:
        if c not in test_data.columns:
            del train_data[c]
        
    for c in test_data.columns[:-1]:
        if c not in train_data.columns:
            del test_data[c]

    return train_data, test_data



def extract_top_k(train_data, k, data='train'):
    """_summary_
    TOP K 개 를 뽑아 Item 번호를 추출
    movieLens의 원래 데이터는 최소 20개, val 데이터는 최소 4개의 데이터를 가진다.

    Args:
        train_data (pandas): 추천할 데이터가 담긴 Data Frame
        k (int): 뽑을 상위 k개 
        data(str) : train data 인지 test 데이터인지에 따라 방법이 다르다.

    Returns:
        list: len(train_data) 행과 k 개의 열을 가진 리스트.
    """

    pred_data_topk = []

    if data=='train':
        for i in range(len(train_data)):
            pred_data_topk.append(train_data.sort_values(by=i, axis=1, ascending=False).columns[:k].tolist())

    else:
        for i in range(1, len(train_data)+1):
            pred_data_topk.append(train_data.iloc[:, :-1].sort_values(by=i, axis=1, ascending=False).columns[:k].tolist())

    return pred_data_topk


def get_NDCG(pred_data, test_data, num, k):
    """_summary_
    그룹 별 추천된 상품목록의 ndcg 를 각각 더하여 
    하나의 ndcg로 도출한다.

    Args:
        pred_data (pandas): 예측한 그룹 별 아이템 추천목록
        test_data (pandas]): 테스트 데이터
        num (int): 그룹 갯수
        k (int): NDCG 의 k 값

    Returns:
        float: result
    """
    
    #각 클러스터 별 결과
    result = [0]*num

    #각 클러스터 별 인원수
    length = list(test_data.cluster.value_counts().sort_index())

    for idx in test_data.index:

        # -1 해당 user 의 그룹을 확인
        cluster_num = int(test_data.loc[idx].cluster) 
        
        # -2 train 에서의 예측치와 test 데이터 간 각 user 별 평점 NDCG 값 계산
        result[cluster_num] += ndcg_score([test_data.loc[idx][:-1]], [pred_data.loc[str(cluster_num)]], k=k) # 일반 검사에서는 str 빼고 그냥 cluster_num 라고 고쳐야 함


    # 최종적으로 total ndcg 를 그룹 내 인원수의 평균 계산 
    for i in range(num):
        result[i] = result[i]/length[i]

    result = sum(result)/len(length)

    #print(f"cluster 수 : {len(length)}\ncluster 별 인원 수 : {length}")
    #print(f"\n총 NDCG : {result:.4f} \n\n ")

    return result


def get_pre_rec(pred_data, test_data, k=3):
    """_summary_
    그룹별 추천된 아이템리스트의 pre, rec 값을 구해서
    하나의 pre, rec 로 나타내는 값

    Args:
        pred_data (pandas): 예측한 그룹 별 아이템 추천목록
        test_data (pandas]): 테스트 데이터
        num (int): 그룹 갯수
        k (int): 정답으로 판단할 평점의 threshold. defualt=3

    Returns:
        float: presicion result
        float: recall    result
    """


    # 1. 전처리
    # 평점이 2.5 이상이면 정답, 아닌 것은 오답으로 평가하였다.
    pred_data[pred_data < k]= 0
    pred_data[pred_data >= k]= 1

    # test 데이터 열에는 cluster 정보가 있기 때문에 백업하는 작업을 거친다.
    tmp = test_data[['cluster']].copy()
    test_data[test_data < k] = 0
    test_data[test_data >= k] = 1
    test_data[['cluster']] = tmp

    # 클러스터의 개수
    cluster_total_num = len(set(test.set_index('cluster').index))

    # 2. 결과값 저장 리스트
    pre_result = [0]*cluster_total_num # 
    rec_result = [0]*cluster_total_num

    length = list(test_data.cluster.value_counts().sort_index())

    ## 4. 각 결과 값에 precision, recall더해줌
    for idx in test_data.index:
        cluster_num = int(test_data.loc[idx].cluster)
        pre_result[cluster_num] += precision_score(list(test_data.loc[idx][:-1].values) ,list(pred_data.loc[cluster_num].values ), average = 'binary')
        rec_result[cluster_num] += recall_score(list(test_data.loc[idx][:-1].values) ,list(pred_data.loc[cluster_num].values ), average = 'binary')

    for i in range(6):
        # 클러스터 별 유저 PRE/REC 의 평균
        pre_result[i] = pre_result[i]/length[i]
        rec_result[i] = rec_result[i]/length[i]


    presicion = sum(pre_result)/len(length)
    recall = sum(rec_result)/len(length)

    # print(f"cluster 수 : {len(length)}\ncluster 별 인원 수 : {length}")
    print(f"Presicion : {presicion:.4f}")
    print(f"\nrecall : {recall:.4f}")

    return presicion, recall 







from multiprocessing.spawn import import_main_path
# 참고자료 : @copyright by https://github.com/Jaehyung-Lim
# Precision Recall의 Ground Truth는 5점으로 설정했다. (근데 우리는 좀 낮출 예정)
def Precision_recall(pred_data, test, k,top=False):

    '''_summary_

        Args:
            pred_data: 결국 개인에게 추천한 것이랑 같게 되는 것 
            test: test dataset #마지막 칼럼은 클러스터링 데이터가 저장된다.
            k: 몇개 추천하는 Precision인지 
            threshold는 일반적으로 5점으로 제한한다.
            top: 이미 추출을 해놨는지에 대한 정보

        Returns:
            precision 값 (float)

    '''

    # 각 클러스터 별 회원 수 
    length = list(test.cluster.value_counts().sort_index()) 
    
    # 클러스터의 개수
    cluster_total_num = len(set(test.set_index('cluster').index))

    # 2. 결과값 저장 리스트
    pre_result = [0]* cluster_total_num
    rec_result = [0]* cluster_total_num
    ## 4. 각 결과 값에 precision, recall더해줌
    for idx in test.index:
        cluster_num = int(test.loc[idx].cluster) #해당 클러스터의 번호

        #train  의 test그룹 번호에 따른 상위 k개 #idx-1

        if not top:
            top_train = list(pred_data.sort_values(by=str(int(test.iloc[idx-1].cluster)), axis=1, ascending=False).columns[:k]) # nash 아닐때는 by = int(test.iloc[idx-1].cluster)
        else:
            top_train = list(pred_data)

        #test 의 상위 k개
        top_test = list(test.iloc[:, :-1].sort_values(by=idx, axis=1, ascending=False).columns[:k])

        #예측을 맞다고 한 대상 pred 중에 실제 test 안에 들어있는 데이터의 비율
        pre_result[cluster_num] +=  len(list(set(top_train) & set(top_test))) / len(top_train)

        #실제 값이 positive 인 대상 중에 예측과 실제 값이 poitive로 일치한 데이터의 비율
        rec_result[cluster_num] +=  len(list(set(top_train) & set(top_test))) / len(top_train)

    for i in range(cluster_total_num):
        # 클러스터 별 유저 PRE/REC 의 평균
        pre_result[i] = pre_result[i]/length[i]
        rec_result[i] = rec_result[i]/length[i]


    presicion = sum(pre_result)/len(length)
    #recall = sum(rec_result)/len(length)

    # print(f"cluster 수 : {len(length)}\ncluster 별 인원 수 : {length}")
    #print(f"Presicion : {presicion:.4f}")
    #print(f"\nrecall : {recall:.4f}")

    return presicion #recall 



### truncated SVD
from predSVD import *
def predict_trunSVD(Ratings):
    """_summary_
    truncated Svd 로 채우지 않은 별점 예측

    Args:
        Ratings (_type_): 기존 user-item maxtirx (Nan값 존재)

    Returns:
        pandas: NaN 값이 SVD 로 예측된 상태 
    """
    
    #평균으로 보간
    rating_table = Rating_filled_in(Ratings)
    #행별 평균으로 정규화
    rating_table, row_mean_data = Rating_norm(rating_table) 
    #특잇값 12로 분해
    rating_table = Rating_svd(rating_table)
    #정규화 해제
    preprocessed_rating = Rating_predict(rating_table,row_mean_data)

    return preprocessed_rating



## MF
def matrix_factorization(train_data, steps=300, alpha=0.0002, beta=0.02):
    """_summary_
    MF 기법으로 평점 데이터 예측.

    Args:
        train_data (pandas): 결측치를 0으로 채운 트레인 데이터
        steps (int, optional): iterations. Defaults to 300.
        alpha (float, optional): 학습률. Defaults to 0.0002.
        beta (float, optional): regularization parameter. Defaults to 0.02.

    Returns:
        _type_: _description_
    """

    #fillna 0 꼭 해야함 -> 무슨 소리 안해도 됨 ;; 
    R = train_data.to_numpy()

    # N: num of user
    N = len(R)
    # M: num of Movie
    M = len(R[0])

    # K: 섞는 비율
    K = 5  #이거를 [5, 10, 15, 20, 25, 30] 정도로 두고 비교하면 좋음
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    # calculate
    Q = Q.T

    start = time.time()
    for step in range(steps):
        print('epoch: %d, time: %f'%(step, time.time()-start))
        for i in range(len(R)):
            for j in range(len(R[0])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)
        e = 0

        for i in range(len(R)):
            for j in range(len(R[0])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    result = P@Q

    return result


