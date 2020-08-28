# inputs
import numpy as np
import scipy.spatial.distance as scidist

class KMeans():
    def __init__(self, clusters):
        self.centers = clusters
        self.centroid_i = []
        self.centroids_ = []
        self.X = None
        self.labels_ = None

    def set_initial_centroids(self):
        # Select three random points from X
        i = 0
        while i < self.centers:
            rand = np.random.randint(len(self.X))
            if i != 0:
                if rand != self.centroid_i[i-1]:
                    self.centroid_i.append(rand)
                    i += 1
            else:
                self.centroid_i.append(rand)
                i += 1

        for i in self.centroid_i:
            centroid_ = []
            for j in range(len(self.X[0])):
                centroid_.append(self.X[i][j])
            self.centroids_.append(centroid_)

    def get_centroids(self, X, nearest_centroids):
        sums = []
        counts = []
        averages = []
        for i in range(len(X[0])):
            sum = [0] * self.centers
            count = [0] * self.centers
            
            for j in range(len(self.X)):
                sum[nearest_centroids[j]] += float(self.X[j][i])
                count[nearest_centroids[j]] += 1
            sums.append(sum)
            counts.append(count)

        for i in range(len(counts)):
            average = []
            for j in range(len(counts[0])):
                average.append((sums[i][j]) / (counts[i][j]))
            averages.append(average)

        averages = list(map(list, zip(*averages)))

        return averages

    def find_nearest_centroid(self, df, centroids):
        last_centroids = [np.random.choice([0,1,2])] * len(X)
        i = 0
        centroids = self.centroids_.copy()
        while True:
            if i>0:
                centroids = self.get_centroids(X, last_centroids)
            distances = scidist.cdist(X, centroids)
            nearest_centroids = np.argmin(distances, axis=1)
            
            if (list(nearest_centroids) == list(last_centroids)):
                centroids_ = centroids
                return nearest_centroids
            else:
                i += 1
                last_centroids = nearest_centroids
        

    def fit(self, X):
        self.X = X
        self.set_initial_centroids() #this will initialize self.centroids_
        self.labels_ = self.find_nearest_centroid(self.X, self.centroids_) #this will assign a centroid to each value of X
           
    def predict(self, X):
        distances = scidist.cdist(X, self.centroids_)
        mins = []
        mins_i = []
        for i in distances:
            minimum = np.min(i)
            mins.append(minimum)
            mins_i.append(np.where(i == minimum)[0][0])
        return mins_i

    def print_centroids(self):
        for i in self.centroids_:
            print(i)


if __name__ == '__main__':

    # Sample set
    #1: count = 250, 3 clusters, 2 features
    
    # Read data from '250_3cl_2feat.csv' into X
    with open('250_3cl_2feat.csv','r') as file:
        X = []
        for i, line in enumerate(file):
            # print(line)
            if i != 0:
                words = line.split(',')
                row = [words[1]]
                row.append(words[2])
                X.append(row)


    # Display first five rows of X:
    print('\n'*2, "*** First five rows of X ***", X[:10], '\n')

    # print([round(j,4) for i in i in X[:10]])
    print([i for i in X[:10]])

    # Create new KMeans object
    kmeans = KMeans(3)

    # Fit the model: assigns each point in X to a cluster
    kmeans.fit(X)
    print("*** Centroids ***",kmeans.print_centroids(),'\n')

    print("*** Clusters ***", kmeans.labels_, '\n')
    
    sample = [[6,9],[6,11]]
    print(f"*** Prediction for {sample} *** \n", kmeans.predict([[6,9],[6,11]]),'\n')



        
