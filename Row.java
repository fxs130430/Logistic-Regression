package hw2;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;


public class Row 
{
    public HashMap<String,Integer> hmap;
    private int Label;
    
    public Row()
    {
        hmap = new HashMap<>();
    }
    public int getDimension()
    {
        return hmap.size();
    }
    public void put(String feature,int val)
    {
        hmap.put(feature, val);
    }
    public void setLabel(int label)
    {
        Label = label;
    }
    public int getLabel()
    {
        return Label;
    }
    public int get(String feature)
    {
        if(!hmap.containsKey(feature))
        {
            System.out.printf("Invalid Feature %s\r\n", feature);
            System.exit(-1);
        }
        return hmap.get(feature);
    }
    
    public void PrintRow()
    {
        Iterator it = hmap.entrySet().iterator();
        while(it.hasNext())
        {
            Map.Entry pair = (Map.Entry)it.next();
            //if((int)pair.getValue() > 0)
            System.out.printf("%s\t%d \r\n", (String)pair.getKey(),(int)pair.getValue());
        }
    }
}
