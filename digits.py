from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

CPU = 12


def main():
    print ("Reading training set")
    dataset = genfromtxt(open('./data/train.csv', 'r'), delimiter=',', dtype='int64')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    print ("Reading test set")
    test = genfromtxt(open('./data/test.csv', 'r'), delimiter=',', dtype='int64')[1:]

    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=CPU)
    print ("Fitting RF classifier")
    rf.fit(train, target)

    print ("Predicting test set")
    predict = rf.predict(test)
    savetxt('submission-version-1.csv', predict, delimiter=',', fmt='%d')
    
    total = len(predict)
    with open('random_forest_submission.csv', 'w') as file:
        file.write("ImageId,Label\n")
        for x in range(0, total):
            string = str(x+1)+","+str(predict[x])+"\n"
            file.write(string)
        file.close()
        
    

if __name__ == "__main__":   
    main()