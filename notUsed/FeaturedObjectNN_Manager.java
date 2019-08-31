package machineLearning;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import de.uos.fmt.musitech.utility.general.WrongArgumentException;
import fuzzy.training.FPTrainingExample;
import fuzzy.training.algorithms.RProp;

import machineLearning.RelativeTrainingExample;

import matlab.MatlabNN;
import matlab.MatlabRowObj;

public class FeaturedObjectNN_Manager extends MatlabNN implements NeuralNetworkManagerInterface {

	private static final int NOTE_TO_NOTE = 0;
	private static final int CHORD_TO_CHORD = 1;

	
	public FeaturedObjectNN_Manager(int inSize, int hidNum, int hidSize) {
		super(inSize, hidNum, hidSize);
		// TODO Auto-generated constructor stub
	}
	
	public void setDataConverter(DataConverter argDataConverter) {	
	}
	
	public void setOutputEvaluator(OutputEvaluator argOutputEvaluator) {
	}
		
	public void setErrorCalculator(ErrorCalculator argErrorCalculator) {
	}
	
	public void setVoicesCoDNotes(List<Integer[]> argVoicesCoDNotes) {
	}
	
	public void setIsNewModel(boolean arg) {
	}
	
	
	@Override
	public void setPossibleVoiceAssignmentsAllChords(List<List<List<Integer>>> argPossibleVoiceAssignmentsEntirePiece) {
		
	}

//	@Override
//	public void createTrainingExample(List<Double> inputs, List<Double> outputs) {
//		throw new NotImplementedException();
//	}
	
	@Override
	public void createTrainingExamplesNoteToNote(List<List<Double>> argOnsetFeatures, List<List<Double>> argLabels, 
		boolean isBidirectional, boolean modelDurationAgain) {
		throw new NotImplementedException();
	}

//	@Override
//	public List<Double> getWeights(File file) {
//		throw new NotImplementedException();
//	}
  
	@Override
	public void createTrainingExamplesChordToChord(List<List<List<Double>>> argChordFeatures, 
		List<List<List<Double>>> argChordLabels) {

	}
	

//	@Override
//	public void setTrainingSetPieceNames(List<String> argTrainingSetPieceNames) {
//		
//	}
		
//	@Override
//	public void storeWeights(File file) {
//		try {
//			fp.storeWeights(new FileOutputStream(file));
//		} catch (FileNotFoundException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}

	
	public void setWeightsFile(File argWeightsFile) {
		
	}
	

	public void setWeightsList(List<Double> argWeightsList) {
	
	}
	
	
	@Override
	public void initWeights(double weightsInitialisationMethod) {
		fp.initializeWeightsRandom(0, .1);
	}

	@Override
	public double[] trainNetwork(Map<String, Double> param, boolean fixFlatSpot) {
		throw new NotImplementedException();
	}

	
	@Override
	public void trainMultipleRuns(Map<String, Double> argTrainingInfoMap, String prefix) {
		throw new NotImplementedException();
	}
	
	
//	@Override
//	public void trainMultipleRuns(int learningApproach, int epoch, OutputEvaluator outputEvaluator, 
//		ErrorCalculator errorCalculator, DataConverter dataConverter, String prefix) {
//		if(learningApproach == NOTE_TO_NOTE)
//			throw new NotImplementedException();
//		else if(learningApproach == CHORD_TO_CHORD){
//			for (int i = 0; i < epoch; i++) {
//				fp.training(rteList, algo);
//			}
//		}			
//	}
	

//	@Override
//	public void storeTrainingRecordCurrentRun(Map<String, Double> argTrainingInfoMap, int run, 
//		List<List<Integer>> assignmentErrors,	String prefix, List<Double> bestWeights, 
//		DataConverter argDataConverter,	OutputEvaluator argOutputEvaluator, ErrorCalculator argErrorCalculator) {
//		throw new NotImplementedException();
//	}
	
	@Override
	public void storeTrainingRecordCurrentRun(Map<String, Double> argTrainingInfoMap, int run, 
			List<List<Integer>> assignmentErrors,	List<List<Integer>> bestPredictedVoices, String currentPath,
			List<Double> bestWeights) {
		throw new NotImplementedException();
	}
	

	@Override
	public double[] evalNetwork(List<Double> inputs) {
		MatlabRowObj mro1 = new MatlabRowObj(inputs.toArray(new Double[]{}));			
		try {
			double val = fp.truthValue("Dist", mro1);
			return new double[]{val};
		} catch (WrongArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

//	@Override
//	public void printOutTrainingInformation() {
//		// TODO Auto-generated method stub
//
//	}

	@Override
	public void divideTrainingAndTestSets(int num) {
		// TODO Auto-generated method stub

	}

	@Override
	public void createCVSets(List<List<List<Double>>> inputs, List<List<List<Double>>> outputs) {
		// TODO Auto-generated method stub

	}

	
	public void setChordSizes(List<Integer> argChordSizes) {
		
	}
	
	
	public void setEqualDurationUnisonsInfo(List<Integer[]> argEqualDurationUnisonsInfo) {
		
	}
	
	
	public void setAllChordOnsetProperties(List<List<List<Integer>>> argAllChordOnsetProperties) {
		
	}
	
	
  public void setAllBasicTabSymbolPropertiesChord(List<Integer[][]> argAllBasicTabSymbolPropertiesChord) {
		
	}
	
	
	@Override
	public void setRelativeTrainingExamples(List<RelativeTrainingExample> relativeTrainingExamples) {
        rteList = new ArrayList<fuzzy.training.RelativeTrainingExample>();
		for (int i = 0; i < relativeTrainingExamples.size(); i++) {
			MatlabRowObj mro1 = new MatlabRowObj(relativeTrainingExamples.get(i).getBetterVal().toArray(new Double[]{}));			
			MatlabRowObj mro2 = new MatlabRowObj(relativeTrainingExamples.get(i).getWorseVal().toArray(new Double[]{}));			
	        fuzzy.training.RelativeTrainingExample rte;
			try {
				rte = fp.createRelativeTrainingExample("Dist", mro1, mro2);
		        rteList.add(rte);	        
			} catch (WrongArgumentException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}

//	@Override
//	public void setTrainingInfoMap(int learningApproach, int featureSet, double allowCoD, double deviationThreshold,
//		int numberOfRuns, double weightsInitialisationType, int maxMetaCycles, int cycles, double alpha, double lambda,
//		double epsilon, int numberOfTrainingExamples) {
//		this.algo = new RProp();
//		algo.setEpsilon(epsilon);
//		algo.setCycles(cycles);
//		algo.setNu(lambda);
		
//	}
	
	
	public void setTrainingInfoMap(Map<String, Double> argTrainingInfoMap) {
		
	}
	

	public void storeTrainingInfoMap(int learningApproach, String prefix) {
		
	}
	
	 public void storeTrainingSetPieceNames(String prefix) {
		 
	 }
	

	public Map<String, Double> getStoredTrainingInfoMap(int learningApproach, String prefix) {
		return null;
	}
	
//	@Override
//	public Map<String, Double> getTrainingInfoMap() {
//		return null;
//	}
	

}
