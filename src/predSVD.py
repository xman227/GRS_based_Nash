import pandas as pd
import numpy as np
import math

from sklearn.decomposition import TruncatedSVD


def Rating_filled_in(rating_table):
    for col in range(len(rating_table.columns)):
        
        # 열의 평균을 구한다.
        col_num = [i for i in rating_table.iloc[:,col] if math.isnan(i)==False]
        col_mean = sum(col_num)/len(col_num)
        
        # NaN을 가진 행은 위에서 구한 평균 값으로 채워준다.
        col_update = [i if math.isnan(i)==False else col_mean for i in rating_table.iloc[:,col]]
        rating_table.iloc[:,col] = col_update
        
    return rating_table


# 2. 각 사용자 별 평균을 빼주며 정규화
def Rating_norm(rating_table):
    
    #추후에 다시 더하기 위해 각 행별 평균을 저장
    row_mean_data = []

    for row in range(len(rating_table)):
        

        # 행의 평균을 구한다.
        row_mean= sum(rating_table.iloc[row,:])/len(rating_table.iloc[row,:])
        
        # 1행부터 마지막행까지의 평균 데이터 저장  
        row_mean_data.append(row_mean)

        # 해당 행의 모든 값에 행 평균 값을 뺀다.
        row_update = [i - row_mean for i in rating_table.iloc[row,:]]
        rating_table.iloc[row,:] = row_update

    return rating_table, row_mean_data
    
def Rating_svd(rating_table):
    # TruncatedSVD를 사용해서 차원축소 ( n_iter : 랜덤 SVD 계산 반복횟수 )
    svd = TruncatedSVD(n_components=12, n_iter=5)
    svd.fit(np.array(rating_table))

    # 특잇값 분해된 행렬 U, S, V
    U     = svd.fit_transform(np.array(rating_table))
    Sigma = svd.explained_variance_ratio_
    VT    = svd.components_

    #print(f'추출된 sigma의 feature : \n {Sigma} \n')

    # Sigma 제곱근처리 후 유사행렬 도출
    ratings_reduced= pd.DataFrame(np.matmul(np.matmul(U, np.diag(Sigma)), VT))

    return ratings_reduced 



def Rating_predict(ratings_table, row_mean_data):
    for row in range(len(ratings_table)):
        # 해당 행의 모든 컬럼 값에 행 평균 값을 더한다.
        ratings_table.iloc[row,:] = row_mean_data[row] + ratings_table.iloc[row,:]
    return ratings_table