/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;
import weka.*;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Geraldi Dzakwan
 */
public class TucilWeka {

    public void loadAndSaveData() throws FileNotFoundException {
        try {
            String filepath = "C:/Users/ASUS/Documents/NetBeansProjects/TucilWeka/src/tucilweka/iris.arff";
            Instances newDataSet = new Instances(new BufferedReader(new FileReader(filepath)));
            System.out.println(newDataSet.toSummaryString());
        } catch (IOException ex) {
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
            test.loadAndSaveData();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
