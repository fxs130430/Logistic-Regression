package hw2;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

public class LogisticRegression {
    
    
    private HashMap<String,Integer> vocabulary;
    private ArrayList<Row> dataset_train;
    private ArrayList<Row> dataset_test;
    private double bias;
    private double[] Weights;
    private final double L_Rate = 0.001;
    private double Lambda;
    
    private double loglikelihood_old = 0;
    
    private double[] Weights_old;
    private double   bias_old;
    
    private String[] feature_names;
    
    private ArrayList<String> stopping_words;
    
    
    public LogisticRegression() 
    {
        vocabulary = new HashMap<>();
        dataset_train = new ArrayList<>();
        dataset_test = new ArrayList<>();
        stopping_words = new ArrayList<>();       
    }
    public void loadDataFromFiles()
    {
        loadStoppingwords();
        createVocabulary();
        HashMap<String,Integer> tempDic = new HashMap<>();
        Iterator it = vocabulary.entrySet().iterator();
        while(it.hasNext())
        {
            Map.Entry pair = (Map.Entry)it.next();
            if((int)pair.getValue() > 40 /* && !stopping_words.contains(pair.getKey())*/)
                tempDic.put((String)pair.getKey(), (Integer)pair.getValue());
        }
        vocabulary = tempDic;
        
        File folder = new File("train/ham");
        File[] listOfFiles = folder.listFiles();
        for(File f:listOfFiles)
            loadFileIntoDataset(dataset_train, f, 0);
        
        folder = new File("train/spam");
        listOfFiles = folder.listFiles();
        for(File f:listOfFiles)
            loadFileIntoDataset(dataset_train, f, 1);
        
        folder = new File("test/ham");
        listOfFiles = folder.listFiles();
        for(File f:listOfFiles)
            loadFileIntoDataset(dataset_test, f, 0);
        
        folder = new File("train/spam");
        listOfFiles = folder.listFiles();
        for(File f:listOfFiles)
            loadFileIntoDataset(dataset_test, f, 1);
        
        
        
        Weights = new double[vocabulary.size()];
        Weights_old = new double[vocabulary.size()];
        bias = 0;
        for(int i = 0 ; i < vocabulary.size() ; i++)
        {
            Weights[i] = 0;
            Weights_old[i] = -1;
            bias_old = -1;
        }
        
        
        
        feature_names = new String[vocabulary.size()];
        it = vocabulary.entrySet().iterator();
        int i = 0;
        while(it.hasNext())
        {
            Map.Entry pair = (Map.Entry)it.next();
            feature_names[i++]= (String)pair.getKey();
            System.out.printf("%s,%d\r\n", pair.getKey(),pair.getValue());
        }
    }
    public void DoTrain(double dLambda)
    {
        
        Lambda = dLambda;
        int nIter = 0;
        while(MoreIterationNeeded())
        {
            gradientAscentIteration();
            
            System.out.printf("Iteration %d [%f]\r\n", ++nIter,getConditionalLogLikehood());
        }
        System.out.println("Converged... :)");
    }
    
    public int predict(Row r)
    {
        double dotProd = dotProduct(r);
        double p_1 = 1 - (1 / (1 + Math.exp(bias + dotProd)));
        int nRet = 1;
        if(p_1 < 0.5)
            nRet = 0;
        return nRet;
    }
    public double getAccuracyOnTestData()
    {
        double errors = 0;
        for(Row r:dataset_test)
        {
            if(r.getLabel() != predict(r))
                errors++;
        }
        double accuracy = 1 - (errors / dataset_test.size());
        return accuracy;
    }
    
    private void gradientAscentIteration()
    {
        loglikelihood_old = getConditionalLogLikehood();
        for(int i = 0 ; i < Weights.length; i++)
        {
            Weights_old[i] = Weights[i];
            bias_old = bias;
        }
        for(int i = 0 ; i < Weights.length ; i++)
        {
            double sum = 0;
            double p = 0;
            int kk = 0;
            for(Row r:dataset_train)
            {
                double X_i = r.get(feature_names[i]);
                double Y = r.getLabel();
                double dotProd = dotProduct(r);
                
                //p(Y = 1|X)

                p = 1 - (1 / (1 + Math.exp(bias + dotProd)));
                sum += X_i * (Y - p);
                
                if(Double.isNaN(sum))
                {
                    System.out.println("oops!");
                    System.exit(-1);
                }
            }
            Weights[i] = Weights[i] + L_Rate * sum - (L_Rate * Lambda * Weights[i]);
            //System.out.printf("i = %d\r\n",i);
        }
        double sum = 0;
        for(Row r:dataset_train)
        {
            int Y = r.getLabel();
            double dotProd = dotProduct(r);
            //p(Y = 1|X)
            double p = 1 - (1 / (1 + Math.exp(bias + dotProd)));
            sum += (Y - p); 
        }
        bias = bias + L_Rate * sum - (L_Rate * Lambda * bias);
    }
    private double norm2(double[] weights)
    {
        double sum = 0;
        for(double d:weights)
            sum += d*d;
        return sum;
    }
    private double getConditionalLogLikehood()
    {
        double sum = 0;
        for(Row r:dataset_train)
        {
            int Label = r.getLabel();
            double dotProd = dotProduct(r);
            double p_1 = 1 - (1 / (1 + Math.exp(bias + dotProd)));
            double p = p_1;
            if(Label == 0)
                p = 1 - p_1;
            if(p == 0)
            {
                sum = -1 ;
                return sum;
            }
            
            sum += Math.log(p);
        }
        sum -= 0.5 * Lambda * norm2(Weights);
        return sum;
    }
    private boolean MoreIterationNeeded()
    {
        boolean bMoreIterationNeeded =false;
        double loglikelihood_new = getConditionalLogLikehood();
        
        if(Math.abs(loglikelihood_new - loglikelihood_old) > 0.001)
            bMoreIterationNeeded = true;
        return bMoreIterationNeeded;
    }
    private double dotProduct(Row r)
    {
        double sum = 0;
        for(int i = 0 ; i < feature_names.length ; i++)
        {
            double X_i = r.get(feature_names[i]);
            if(X_i > 0)
                X_i = 1;
            sum += Weights_old[i] * X_i;
        }
        return sum;
    }
    
    private void createVocabulary()
    {
        File folder = new File("train/ham");
        File[] listOfFiles = folder.listFiles();
        
        loadIntoVocabulary(listOfFiles);
                
        folder = new File("train/spam");
        listOfFiles = folder.listFiles();
        
        loadIntoVocabulary(listOfFiles);
        
        
        folder = new File("test/ham");
        listOfFiles = folder.listFiles();
        
        loadIntoVocabulary(listOfFiles);
                
        folder = new File("test/spam");
        listOfFiles = folder.listFiles();
        
        loadIntoVocabulary(listOfFiles);
        Iterator it = vocabulary.entrySet().iterator();
    }
    private void loadIntoVocabulary(File[] files)
    {
        for(File f: files)
        {
            if(f.isFile())
            {
                String url = f.getAbsolutePath();
                List<String> temp_list = getDistinctWordList(url);
                for(String w: temp_list)
                {
                    if(vocabulary.containsKey(w))
                    {
                        int currentCount = vocabulary.get(w);
                        currentCount++;
                        vocabulary.put(w, currentCount);
                    }
                    else 
                    {
                        vocabulary.put(w, 1);
                    }
                }
            }            
        }
    }
    private List<String> getDistinctWordList(String fileName){
 
        FileInputStream fis = null;
        DataInputStream dis = null;
        BufferedReader br = null;
        List<String> wordList = new ArrayList<String>();
        try {
            fis = new FileInputStream(fileName);
            dis = new DataInputStream(fis);
            br = new BufferedReader(new InputStreamReader(dis));
            String line = null;
            while((line = br.readLine()) != null){
                StringTokenizer st = new StringTokenizer(line, " ,.;:\"");
                while(st.hasMoreTokens()){
                    String tmp = st.nextToken().toLowerCase();
                    if(!wordList.contains(tmp)){
                        wordList.add(tmp);
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally{
            try{if(br != null) br.close();}catch(Exception ex){}
        }
        return wordList;
    }
    private void loadStoppingwords()
    {
        FileInputStream fis = null;
        DataInputStream dis = null;
        BufferedReader br = null;
        try
        {
            fis = new FileInputStream("stoppingwords.txt");
            dis = new DataInputStream(fis);
            br = new BufferedReader(new InputStreamReader(dis));
            String line = null;
            while((line = br.readLine()) != null)
            {
                stopping_words.add(line);
            }
            br.close();
        }
        catch(Exception ex)
        {
            ex.printStackTrace();
        }
    }
    private HashMap<String,Integer> createHashMapFromFile(String URL)
    {
        FileInputStream fis = null;
        DataInputStream dis = null;
        BufferedReader br = null;
        HashMap<String, Integer> wordCount = new HashMap<>();
        try {
            fis = new FileInputStream(URL);
            dis = new DataInputStream(fis);
            br = new BufferedReader(new InputStreamReader(dis));
            String line = null;
            while((line = br.readLine()) != null){
                StringTokenizer st = new StringTokenizer(line, " ,.;:\"");
                while(st.hasMoreTokens()){
                    String tmp = st.nextToken().toLowerCase();
                    if(!wordCount.containsKey(tmp)){
                        wordCount.put(tmp,1);
                    }
                    else
                    {
                        int currentCount = wordCount.get(tmp);
                        currentCount++;
                        wordCount.put(tmp, currentCount); 
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally{
            try{if(br != null) br.close();}catch(Exception ex){}
        }
        return wordCount;
    }
    
    private void loadFileIntoDataset(ArrayList<Row> dataset,File f,int label)
    {
        String url = f.getAbsolutePath();
        HashMap<String,Integer> WC = createHashMapFromFile(url);
        Row r = new Row();
        r.setLabel(label);
        Iterator it = vocabulary.entrySet().iterator();
        while(it.hasNext())
        {
            Map.Entry pair = (Map.Entry)it.next();
            String feature = (String)pair.getKey();
            if(WC.containsKey(feature))
                r.put(feature, WC.get(feature));
            else
                r.put(feature, 0);
        }
        dataset.add(r);  
        //r.PrintRow();
    }    
    
}
