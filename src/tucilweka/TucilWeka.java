/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.Scanner;

/**
 *
 * @author Geraldi Dzakwan
 */
public class TucilWeka {
    
    DataSource dataSource;
    Instances inputDataSet;
    Instances outputDataSet;

    public void loadData() throws FileNotFoundException {
        String loadFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/iris.arff";
        try {
            dataSource = new DataSource(loadFilePath);
            inputDataSet = dataSource.getDataSet();
            //Ini kalo instances langsung
            //Instances inputDataSet = new Instances(new BufferedReader(new FileReader(loadFilePath)));            
            System.out.println(inputDataSet.toSummaryString());
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void saveData() {
        ArffSaver aSaver = new ArffSaver();
        aSaver.setInstances(outputDataSet);
        String saveFilePath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/saved.arff";
        try {
            aSaver.setFile(new File(saveFilePath));
            aSaver.writeBatch();
        } catch (IOException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void discretizeData() {
        String s1 = "", s2 = "", s3 = "", s4 = "", s5 = "";
        Scanner sc = new Scanner(System.in);
        
        /*
        System.out.print("-B, -M , or -O : ");
        s1 = sc.nextLine();
        */
        s1 = "-B";
        
        if ("-B".equals(s1)) {
            System.out.print("How many bins : ");
            s2 = sc.nextLine();
        }
        
        System.out.print("List of columns : ");
        s3 = sc.nextLine();
                
        /*
        System.out.print("-F (yes/no) : ");
        String s4 = sc.nextLine();
        
        System.out.print("-V (yes/no) : ");
        String s5 = sc.nextLine();
        */
        
        int numberOfSettings;
        if("-B".equals(s1)) {
            numberOfSettings = 4;
        } else {
            numberOfSettings = 3;
        }
        
        String[] settings = new String[numberOfSettings];
        
        if ("-B".equals(s1)) {
            settings[0] = s1;
            settings[1] = s2;
            settings[2] = "-R";
            settings[3] = s3;
        } else {
            
        }
        
       
        
        Discretize disc = new Discretize();
        try {
            disc.setOptions(settings);
            disc.setInputFormat(inputDataSet);
            outputDataSet = Filter.useFilter(inputDataSet, disc);
        } catch (Exception ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
       
    }
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        System.out.println("import working");
        TucilWeka test = new TucilWeka();
        try {
            test.loadData();
            test.discretizeData();
            test.saveData();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
