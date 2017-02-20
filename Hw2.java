package hw2;
public class Hw2 {

    public static void main(String[] args) 
    {
        NaiveBayes nb = new NaiveBayes();
        nb.loadDataFromFiles();
        double accuracy = nb.getAccuracyOnTestData();
        System.out.printf("Accuracy = %f\r\n", accuracy);
        
        /*
        LogisticRegression lr = new LogisticRegression();
        lr.loadDataFromFiles();
        lr.DoTrain(0.1);
        double accuracy = lr.getAccuracyOnTestData();
        System.out.printf("Accuracy = %f\r\n", accuracy);
        */
    }
    
}
