package machineLearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.flat.train.prop.RPROPType;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;


public class NNManager { // implements NNManagerInterface {
//	public final static int MAX_NUM_VOICES_N2N = 5;  	
//	private BasicMLDataSet dataset;
//	MLDataPair dataPair;
//	private List<List<Double>> noteFeatures;
//	private List<List<List<Double>>> chordFeatures;

	private final static boolean DEBUG = true;
	private BasicNetwork network; // N2N and C2C
	
	// Ground truth lists - same in test and application mode
//	private List<List<Double>> groundTruthVoiceLabels; // N2N and C2C
//	private List<List<Double>> groundTruthDurationLabels; // N2N	
//	private List<Integer[]> equalDurationUnisonsInfo; // N2N and C2C
//	private List<Integer[]> voicesCoDNotes; // N2N
//	private List<Integer> chordSizes; // N2N
//	private List<List<List<Double>>> groundTruthChordVoiceLabels; // C2C

	// Lists not the same in test and application mode
//	private List<List<List<Integer>>> possibleVoiceAssignmentsAllChords; // C2C
	
//	private List<RelativeTrainingExample> relativeTrainingExamples; 
//	private List<List<List<Integer>>> allChordOnsetProperties = new ArrayList<List<List<Integer>>>(); 
//	private List<Integer[][]> allBasicTabSymbolPropertiesChord = new ArrayList<Integer[][]>();
//	public List<String> trainingSetPieceNames;
//	static int callCount = 0;
//	private boolean modelDuration; // in training mode switched on through createTrainingExamplesNoteToNote(); in test/application mode via TestManager.startTestProcess()
//	private boolean isBidirectional; // in training mode switched on through createTrainingExamplesNoteToNote(); in test/application mode switched on via TestManager.startTestProcess()
//	private boolean modelDurationAgain; // in training mode switched on through createTrainingExamplesNoteToNote(); in test/application mode switched on via TestManager.startTestProcess()
//	private boolean isNewModel; //2016
//	private boolean storeAdditionalFiles;
//	private List<String> trainingPieceNames;
//	private String trainingSettings;
//	private Map<String, Double> trainingParameters;

//	private OutputEvaluator outputEvaluator;
//	private ErrorCalculator errorCalculator;
//	private DataConverter dataConverter;
//	private ToolBox toolBox;

//	public enum WeightsInit{RANDOM, FROM_LIST, FROM_FILE};  
//	public static final int RANDOM_INIT = 0; 
//	public static final int INIT_FROM_LIST = 1;
//	public static final int INIT_FROM_FILE = 1;
//	public static final int NOTE_CLASS = 0;
//	public static final int CHORD_REGRESS = 1;
	
	public static ActivationFunction[] ALL_ACT_FUNCT = new ActivationFunction[ActivationFunction.values().length];
	static {
		Arrays.asList(ActivationFunction.values()).forEach(af -> ALL_ACT_FUNCT[af.getIntRep()] = af);
	}
	public static enum ActivationFunction {
		SIGMOID("sgm", 0), SOFTMAX("sft", 1), RELU("ReLU", 2);
		
		private int intRep;
		private String stringRep;
		ActivationFunction(String s, int i) {
			this.stringRep = s;
			this.intRep = i;
//			ALL_ACT_FUNCT[i] = this;
		}
		
		@Override
	    public String toString() {
	        return getStringRep();
	    }
		
		public String getStringRep() {
			return stringRep;
		}
		
		public int getIntRep() {
			return intRep;
		}
	};
	
//	private static final int TOLERANT = 0;
//	private static final int STRICT = 1;
	
//	public static final String ACTIVATION_FUNCTION = "activation_function"; 
//	public static final String MODELLING_APPROACH = "modellingApproach";
//	public static final String MODELLING_APPROACH = "modelling_approach";
//	public static final String WEIGHTS_INIT_METHOD = "weightsInitMethod";
//	public static final String WEIGHTS_INIT_METHOD = "weights_init_method";
//	public static final String MAX_META_CYCLES = "maxMetaCycles";
//	public static final String MAX_META_CYCLES = "max_meta_cycles";
	public static final String CYCLES = "cycles";
//	public static final String LEARNING_RATE = "learningRate";
	public static final String LEARNING_RATE = "learning rate";
//	public static final String REGULARISATION_PARAMETER = "regularisationParameter";
	public static final String REGULARISATION_PARAMETER = "regularisation parameter";
//	public static final String HIDDEN_NEURONS_FACTOR = "hiddenNeuronsFactor";
//	public static final String HIDDEN_NEURONS_FACTOR = "hidden_neurons_factor";
	public static final String MARGIN = "margin";
	public static final String ACT_FUNCTION = "activation function";
	

//	public static final int NOTE_TO_NOTE = 0;
//	public static final int CHORD_TO_CHORD = 1;
//	public static final double COD_NOT_ALLOWED = 0.0;
//	public static final double COD_ALLOWED  = 1.0;
//	public static enum LearningApproach {NOTE_TO_NOTE, CHORD_TO_CHORD};
//	public static final String LEARNING_APPROACH = "learningApproach";
////	public static final String FEATURE_SET = "feature_set";
////	public static final String ERROR_MEASUREMENT = "error_measurement";
//	public static final String ALLOW_COD = "allowCoD";
//	public static final String DEVIATION_THRESHOLD = "deviationThreshold";
////	public static final String NUMBER_OF_RUNS = "num_runs";
////	public static final String NUMBER_OF_TRAINING_EXAMPLES = "num_training_ex";
//	public static final String HIGHEST_NUMBER_OF_VOICES = "highestNumVoices";
//	public static final String LARGEST_CHORD_SIZE = "largestChordSize";
//	public static final String MODEL_DURATION = "modelDuration";
//	public static final String MODEL_BACKWARD = "modelBackward";
//	public static final String MODEL_MELODY = "modelMelody";
//	public static final String IS_BIDIRECTIONAL = "isBidirectional";
//	public static final String MODEL_DURATION_AGAIN = "modelDurationAgain";
	
//	private List<Double> weightsList;
//	private File weightsFile;
  
//	private String directory = ""; // directory for data to be read and stored
//	public static String bestWeightsFileName = "best_weights";
//	private String bestWeightsFileNoteToNotePrefix = "bestWeights (N)";
//	private String bestWeightsFileChordToChordPrefix = "bestWeights (C)";
//	public static String initWeightsFileName = "init_weights";
//	public static String initialWeightsFileNoteToNotePrefix = "initialWeights (N)";
//	public static String initialWeightsFileChordToChordPrefix = "initialWeights (C)";
//	public static String bestWeightsNoteToNoteSuffix = "bestWeights (N).xml";
//	public static String bestWeightsChordToChordSuffix = "bestWeights (C).xml";
//	public static String trainingInfoMapNoteToNoteSuffix = "trainingParameters (N).xml";
//	public static String trainingInfoMapChordToChordSuffix = "trainingParameters (C).xml";
//	public static String trainingSetPieceNamesSuffix = "Training set pieces names.xml";
//	public static String numTrainingEx = "num_training_ex";
//	public static String numberOfTrainingExamplesNoteToNoteSuffix = "numTrainingExamples (N).xml";
//	public static String numberOfTrainingExamplesChordToChordSuffix = "numTrainingExamples (C).xml";

//	public static String trainRec = "training_rec";
//	public static String testRec = "test_rec_";
//	public static String applRec = "application_rec_";
//	public static String headerRec = "MODEL PARAMETERS AND DATA";
//	public static String specRec = "RESULTS";
//	public static String detailsRec = "GROUND TRUTH VOICES AND PREDICTED VOICES DETAILS";
//	public static String detailsRec = "DETAILS";
	
//  private Map<String, Double> trainingInfoMap = new HashMap<String, Double>();
//	private String trainingStartTime;
//	private String trainingEndTime;
//	private String testStartTime;
//	private String testEndTime;
	
	public enum Comparator {SGM, SLN, AVG};
	private static Comparator COMPARATOR = Comparator.SLN;
	private double gamma = 300;
	
//	private boolean SIGMOID_COMPARATOR = true;
//	private double gamma = 300;
//	private double nu;


//	/**
//	 * Create a new network with the given number of inputs, a hidden layer of 2 times the size of the input
//	 * layer, and an output layer with the given number of outputs. 
//	 * 
//	 * @param numFeatures The number of input features.
//	 * @param numOutputs The number of outputs.
//	 */
//	public EncogNNManager(int numFeatures, int numOutputs) {
//		network = new BasicNetwork() ;
//		network.addLayer (new BasicLayer (null, true, numFeatures)); // input
//		network.addLayer (new BasicLayer (new ActivationSigmoid(), true, 2*numFeatures)); // hidden
//		network.addLayer (new BasicLayer (new ActivationSigmoid(), false, numOutputs)); // output
//		network.getStructure().finalizeStructure();
//		network.reset();
//	}

	
	public NNManager() {
		
	}
	
	/**
	 * Initialises a new three-layer neural network. 
	 * 
	 * @param actFunc The activation function
	 * @param layerSizes An Integer[] containing <br>
	 *                   as element 0: the number of input neurons <br>
	 *                   as element 1: the number of hidden neurons <br>
	 *                   as element 2: the number of output neurons <br>  
	 *                   Bias neurons are always excluded.
	 */
	public void initialiseNetwork(ActivationFunction actFunc, Integer[] layerSizes) {
		
		int numFeatures = layerSizes[0];
		int numHidden = layerSizes[1];
		int numOutputs = layerSizes[2];
		network = new BasicNetwork();
		// Input layer
		network.addLayer(new BasicLayer(null, true, numFeatures));
		// Hidden layer
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHidden));
		// Output layer
		if (actFunc == ActivationFunction.SIGMOID) {
			network.addLayer(new BasicLayer(new ActivationSigmoid(), false, numOutputs));
		}
		else if (actFunc == ActivationFunction.SOFTMAX) {
			network.addLayer(new BasicLayer(new ActivationSoftMax(), false, numOutputs));
		}
		network.getStructure().finalizeStructure();
		network.reset();
	}
	
	
//	/**
//	 * Create a NeuralNetworkManager with a custom network.
//	 * 
//	 * @param bn The network to use in this manager. 
//	 */
//	public EncogNNManager(BasicNetwork bn) {
//		network = bn;
//	}


//	public void setGroundTruthVoiceLabels(List<List<Double>> argGroundTruthVoiceLabels) {
//		groundTruthVoiceLabels = argGroundTruthVoiceLabels;
//	}

//	public List<List<Double>> getGroundTruthVoiceLabels() {
//		return groundTruthVoiceLabels;
//	}


//	public void setGroundTruthDurationLabels(List<List<Double>> argGroundTruthDurationLabels) {
//		groundTruthDurationLabels = argGroundTruthDurationLabels;
//	}

//	public List<List<Double>> getGroundTruthDurationLabels() {
//		return groundTruthDurationLabels;
//	}


//	public void setEqualDurationUnisonsInfo(List<Integer[]> argEqualDurationUnisonsInfo) {
//		equalDurationUnisonsInfo = argEqualDurationUnisonsInfo;
//	}

//	public List<Integer[]> getEqualDurationUnisonsInfo() {
//		return equalDurationUnisonsInfo;
//	}


//	public void setVoicesCoDNotes(List<Integer[]> argVoicesCoDNotes) {
//		voicesCoDNotes = argVoicesCoDNotes;
//	}

//	public List<Integer[]> getVoicesCoDNotes() {
//		return voicesCoDNotes;
//	}


//	public void setChordSizes(List<Integer> argChordSizes) {
//		chordSizes = argChordSizes;
//	}

//	public List<Integer> getChordSizes() {
//		return chordSizes;
//	}


//	public void setGroundTruthChordVoiceLabels(List<List<List<Double>>> argGroundTruthChordVoiceLabels) {
//		groundTruthChordVoiceLabels = argGroundTruthChordVoiceLabels;
//	}

//	public List<List<List<Double>>> getGroundTruthChordVoiceLabels() {
//		return groundTruthChordVoiceLabels;
//	}


//	public void setPossibleVoiceAssignmentsAllChords(List<List<List<Integer>>> argPossibleVoiceAssignmentsAllChords) {
//		possibleVoiceAssignmentsAllChords = argPossibleVoiceAssignmentsAllChords;
//	}

//	public List<List<List<Integer>>> getPossibleVoiceAssignmentsAllChords() {
//		return possibleVoiceAssignmentsAllChords;
//	}


//	public void setDataConverter(DataConverter argDataConverter) {
//		dataConverter = argDataConverter;
//	}	

//	public void setOutputEvaluator(OutputEvaluator argOutputEvaluator) {
//		outputEvaluator = argOutputEvaluator;
//	}

//	public void setErrorCalculator(ErrorCalculator argErrorCalculator) {
//		errorCalculator = argErrorCalculator;
//	}
	
	
//	public void setToolBox(ToolBox arg) {
//		toolBox = arg;
//	}


//	public void setModelDuration(boolean arg) {
//	  modelDuration = arg;	
//	}

//	public void setIsBidirectional(boolean arg) {
//		isBidirectional = arg;	
//	}

//	public void setModelDurationAgain(boolean arg) {
//		modelDurationAgain = arg;	
//	}

//	public void setIsNewModel(boolean arg) { //2016
//		isNewModel = arg;
//	}

//	public void setStoreAdditionalFiles(boolean arg) {
//		storeAdditionalFiles = arg;
//	}

//	public void setTrainingPieceNames(List<String> arg) {
//		trainingPieceNames = arg;
//	}

//	public void setTrainingSettings(String arg) {
//		trainingSettings = arg;
//	}

//	public void setTrainingParameters(Map<String, Double> arg) {
//		trainingParameters = arg;
//	}

//	private void setDataset(BasicMLDataSet arg) {
//		dataset = arg;
//	}
	
//	public void setNoteFeatures(List<List<Double>> arg) {
//		noteFeatures = arg;
//	}


	private BasicMLDataSet createDataset(List<List<Double>> argNoteFeatures, List<List<Double>> argLabels) {
		double[][] inputs = new double[argNoteFeatures.size()][argNoteFeatures.get(0).size()];
		for (int i = 0; i < argNoteFeatures.size(); i++) {
			for (int j = 0; j < argNoteFeatures.get(i).size(); j++) {
				inputs[i][j] = argNoteFeatures.get(i).get(j);	
			}
		}
		double[][] outputs = new double[argLabels.size()][argLabels.get(0).size()];
		for (int i = 0; i < argLabels.size(); i++) {
			for (int j = 0; j < argLabels.get(i).size(); j++) {
				outputs[i][j] = argLabels.get(i).get(j);	
			}
		}
//		dataset = new BasicMLDataSet(inputs, outputs);
//		setDataset(new BasicMLDataSet(inputs, outputs));
		return new BasicMLDataSet(inputs, outputs);
	}


	/**
	 * Generates, for the entire piece, a List of RelativeTrainingExamples from the given List<List<List<>>> of all 
	 * possible chord feature vectors for each chord in the piece. Each RelativeTrainingExample consists of two elements:
	 * the first will always be the chord feature vector that goes with the ground truth voice assignment (i.e., the first 
	 * one from the List<List<Double>> representing all possible feature vectors for that chord), and the second will be
	 * one of the remaining possibilities, in ascending order. For each chord, this yields a List<RelativeTrainingExample>
	 * that looks like [[fv0, fv1], [fv0, fv2], ... , [fv0, vfn]], where n is the index of the last possible chord feature
	 * vector. For each chord, this List is added to the end of the List<RelativeTrainingExample> that is returned by the 
	 * method after the last chord is dealt with.   
	 * 
	 * @param allCompleteChordFeatureVectors
	 * @return
	 */
	// TESTED
	List<RelativeTrainingExample> generateAllRelativeTrainingExamplesBEFOREMOVE(List<List<List<Double>>>
		allCompleteChordFeatureVectors) {
		List<RelativeTrainingExample> allRelativeTrainingExamples = new ArrayList<RelativeTrainingExample>();

		for (int i = 0; i < allCompleteChordFeatureVectors.size(); i++) {
			List<List<Double>> currentCompleteChordFeatureVectors = allCompleteChordFeatureVectors.get(i);  
			// Get the chord feature vector for the ground truth, which is always the first element of currentCompleteChordFeatureVectors
			List<Double> currentGroundTruthChordFeatureVector = currentCompleteChordFeatureVectors.get(0);
			// Pair currentGroundTruthChordFeatureVector up with all other possible chord feature vectors to form the  
			// RelativeTrainingExamples, and add each of these to relativeTrainingExamples
			for (int j = 1; j < currentCompleteChordFeatureVectors.size(); j++) {
				List<Double> currentChordFeatureVector = currentCompleteChordFeatureVectors.get(j);
				RelativeTrainingExample currentRelativeTrainingExample = 
					new RelativeTrainingExample(currentGroundTruthChordFeatureVector, currentChordFeatureVector);
				allRelativeTrainingExamples.add(currentRelativeTrainingExample);	
			}
		}
		return allRelativeTrainingExamples;
	}


//	/**
//	 * Sets noteFeatures, groundTruthLabels, groundTruthVoiceLabels, groundTruthDurationLabels (if applicable),
//	 * and dataSet. Thus creates a List of training examples out of the given features and labels. Each training
//	 * example in the list is wrapped in an MLDataPair, and the MLDataPairs themselves are wrapped in a 
//	 * BasicMLDataSet.
//	 * 
//	 * NB: NOTE_TO_NOTE approach only.
//	 * 
//	 * @param argNoteFeatures
//	 * @param argLabels
//	 */
//	public void createTrainingExamplesNoteToNote(List<List<Double>> argNoteFeatures, 
//		List<List<Double>> argLabels, boolean argIsBidirectional, boolean argModelDurationAgain) { 
//
//		// Set modelDuration 
//		if (argLabels.get(0).size() > MAX_NUM_VOICES_N2N) {
//			setModelDuration(true);
//		}
//		if (argIsBidirectional) {
//			setIsBidirectional(true);
//		}
//		if (argModelDurationAgain) {
//			setModelDurationAgain(true);
//		}
//
//		// Turn Lists into Arrays 
//		double[][] inputs = new double[argNoteFeatures.size()][argNoteFeatures.get(0).size()];
//		for (int i = 0; i < argNoteFeatures.size(); i++) {
//			for (int j = 0; j < argNoteFeatures.get(i).size(); j++) {
//				inputs[i][j] = argNoteFeatures.get(i).get(j);	
//			}
//		}
//		double[][] outputs = new double[argLabels.size()][argLabels.get(0).size()];
//		for (int i = 0; i < argLabels.size(); i++) {
//			for (int j = 0; j < argLabels.get(i).size(); j++) {
//				outputs[i][j] = argLabels.get(i).get(j);	
//			}
//		}
//
//		// Set noteFeatures, groundTruthVoiceLabels, and, if applicable, groundTruthDurationLabels	
//		noteFeatures = argNoteFeatures;
//		groundTruthLabels = argLabels;
//		groundTruthVoiceLabels = new ArrayList<List<Double>>();
//		if (modelDuration) {
//			groundTruthDurationLabels = new ArrayList<List<Double>>();
//		}
//		for (List<Double> l : argLabels) {
//			groundTruthVoiceLabels.add(new ArrayList<Double>(l.subList(0, MAX_NUM_VOICES_N2N)));
//			if (modelDuration) {	
//				groundTruthDurationLabels.add(new ArrayList<Double>(l.subList(MAX_NUM_VOICES_N2N, l.size())));
//			}
//		}
//		dataset = new BasicMLDataSet(inputs, outputs);
//	}


//	/**
//	 * Sets chordFeatures, groundTruthVoiceLabels, and groundTruthChordVoiceLabels. 
//	 * NB: CHORD_TO_CHORD approach only.
//	 * 
//	 * @param argChordFeatures
//	 * @param argChordLabels
//	 */
//	public void createTrainingExamplesChordToChord(List<List<List<Double>>> argChordFeatures, 
//		List<List<List<Double>>> argChordLabels) {
//		chordFeatures = argChordFeatures;
//		groundTruthChordVoiceLabels = argChordLabels;
//		setGroundTruthVoiceLabels(null, groundTruthChordVoiceLabels);
//		groundTruthVoiceLabels = createGroundTruthVoiceLabels(groundTruthChordVoiceLabels);
//		groundTruthVoiceLabels = new ArrayList<List<Double>>();
//		for (List<List<Double>> l : groundTruthChordVoiceLabels) {
//			groundTruthVoiceLabels.addAll(l);
//		}
//	}


//	/**
//	 * Sets groundTruthVoiceLabels.
//	 * 
//	 * @param argGroundTruthVoiceLabels Is <code>null</code> in C2C case
//	 * @param argGroundTruthChordVoiceLabels Is <code>null</code> in N2N case
//	 */
//	void setGroundTruthVoiceLabels(List<List<Double>> argGroundTruthVoiceLabels, 
//		List<List<List<Double>>> argGroundTruthChordVoiceLabels) {
//		
//		// Verify that one of the arguments is null and the other not
//		if ((argGroundTruthVoiceLabels != null && argGroundTruthChordVoiceLabels != null) || 
//			(argGroundTruthVoiceLabels == null && argGroundTruthChordVoiceLabels == null)) {
//			System.out.println("ERROR: if argGroundTruthVoiceLabels == null, "
//				+ "argGroundTruthChordVoiceLabels must not be, and vice versa" + "\n");
//			throw new RuntimeException("ERROR (see console for details)");
//		}
//		
//		// N2N
//		if (argGroundTruthVoiceLabels != null) {
//			groundTruthVoiceLabels = argGroundTruthVoiceLabels;
//		}
//		// C2C
//		else {
//			groundTruthVoiceLabels = new ArrayList<List<Double>>();
//			for (List<List<Double>> l : argGroundTruthChordVoiceLabels) {
//				groundTruthVoiceLabels.addAll(l);	
//			}
//		}
//	}


//	public void setRelativeTrainingExamples(List<RelativeTrainingExample> argRelativeTrainingExamples) {
//		relativeTrainingExamples = argRelativeTrainingExamples;  
//	}
	
	
//	/**
//	 * Sets allBasicTabSymbolPropertiesChord.
//	 * NB: CHORD_TO_CHORD approach only.
//	 * 
//	 * @param argAllBasicTabSymbolPropertiesChord
//	 */
//	public void setAllBasicTabSymbolPropertiesChord(List<Integer[][]> argAllBasicTabSymbolPropertiesChord) {
//		allBasicTabSymbolPropertiesChord = argAllBasicTabSymbolPropertiesChord;  
//	}


	/**
	 * Trains the network, using the settings as stored in the map. 
	 * 
	 * @param map
	 * @param fixFlatSpot Whether or not to fix the flat spot problem (http://www.heatonresearch.com/wiki/Flat_Spot)
	 * @return An array with 2 values: [the error before training, the error after training].
	 */
	public double[] trainNetwork(Map<String, Double> map, boolean fixFlatSpot, 
		/*BasicMLDataSet argDataset*/ List<double[][]> dataSetPlaceholder) {
				
		int cycles = map.get(CYCLES).intValue();
		double alpha = map.get(LEARNING_RATE);
		double lambda = map.get(REGULARISATION_PARAMETER);

		BasicMLDataSet argDataset = 
			new BasicMLDataSet(dataSetPlaceholder.get(0), dataSetPlaceholder.get(1));
		
		System.out.println("Start training:\n==============");
		System.out.println("Starting the training with " 
//		+ dataset.getRecordCount() + 
		+ argDataset.getRecordCount() + 
		" training example(s) for max " + cycles + " cycles.");                   

//		Backpropagation train = new Backpropagation(network, dataset);
//		train.setLearningRate(alpha);
//		ResilientPropagation train = new ResilientPropagation(network, dataset);
		ResilientPropagation train = new ResilientPropagation(network, argDataset);
		train.setRPROPType(RPROPType.iRPROPp); // was classic RPROP before 10-2-15
		double[] error = new double[2];
		error[0] = train.getError();
		train.fixFlatSpot(fixFlatSpot); 
		train.setThreadCount(1);
		for (int i = 0; i < cycles; i++) {
			// Train once
			train.iteration(1);
			// Regularise
			regularise(alpha * lambda);
		} // end of training loop
		
		error[1] = train.getError();
		System.out.println("Error after training " + error[1] + " in " + cycles + " training cycles. "); 
		return error;
	}

	
	/**
	 * Trains the network according to the given parameters and stores, for each run,
	 * (1) the weights that go with the lowest classification error obtained in that run
	 *     in an .xml file;
	 * 
	 * Runs - metacycles - cycles
	 * Each run, a fresh network that trains anew is created
	 * Each metacycle is a measurement step within the training process, where we determine the training and
	 * classification errors to see how the learning goes
	 * Each cycle is one forward-backward propagation step of the NN over all training examples where we adjust
	 * the weights.
	 * --> 3 runs with 20 metacycles of each 10 cycles means that the network is trained (i.e., the weights are 
	 * changed) 3 * 20 * 10 = 600 times.
	 *  
	 * @param modelParameters
	 * @param path
	 * @param trainingSettings
	 * @param trainingData N2N case: contains two elements: features (index 0) and labels (index 1)
	 *                     C2C case: contains the n chordFeatures, where n is the number of chords 
	 * @param argVoicesCoDNotes Only in N2N case; <code>null</code> in C2C case.
	 */
	// FIXME commented out 12.11.2016 
	private List<List<List<Integer>>> trainMultipleRuns(
		Map<String, Double> modelParameters, /* String info[],*/ 
		String path, 
		/*String trainingSettings, List<Integer[]> argVoicesCoDNotes,*/ 
		List<List<List<Double>>> trainingData 
		/*, List<RelativeTrainingExample> argRelativeTrainingExamples*/
//		, List<List<Double>> argGroundTruthVoiceLabels 
//		, List<List<Double>> argGroundTruthDurationLabels
		, List<List<List<Double>>> argGroundTruths
		, List<Integer[]> argEqualDurationUnisonsInfo
		, List<List<List<Integer>>> argPossibleVoiceAssignmentsAllChords
		) {
//
//		List<List<List<Integer>>> trainingResults = new ArrayList<List<List<Integer>>>();
//		
////		String path = info[1];
////		String trainingSettings = info[2];
//		
////		boolean errorMeasurementIsStrict = false;
////		if (modelParameters.get(ERROR_MEASUREMENT).intValue() == STRICT) {
////			errorMeasurementIsStrict = true;
////		}
//
//		int modellingApproach = modelParameters.get(MODELLING_APPROACH).intValue();
//		int weightsInitialisationMethod = modelParameters.get(WEIGHTS_INIT_METHOD).intValue();
//		int maxMetaCycles = modelParameters.get(MAX_META_CYCLES).intValue();
//
//		BasicMLDataSet trainingSetN2N = null;
//		List<RelativeTrainingExample> argRelativeTrainingExamples = null;
//		if (modellingApproach == NOTE_CLASS) {
//			trainingSetN2N = createDataset(trainingData.get(0), trainingData.get(1));
//		}
//		else if (modellingApproach == CHORD_REGRESS) {
//			argRelativeTrainingExamples = 
//				generateAllRelativeTrainingExamples(trainingData);
//		}
//		
//		List<List<Double>> argGroundTruthVoiceLabels = argGroundTruths.get(0);
//		List<List<Double>> argGroundTruthDurationLabels = argGroundTruths.get(1);
//		
////		List<List<Double>> argGroundTruthVoiceLabels = getGroundTruthVoiceLabels(); 6 maart
////		List<List<Double>> argGroundTruthDurationLabels = getGroundTruthDurationLabels(); 6 maart
////		List<Integer[]> argEqualDurationUnisonsInfo = getEqualDurationUnisonsInfo(); 6 maart
//		
//		boolean fixFlatSpot;
//
//		List<Double> smallestClassificationErrors = new ArrayList<Double>();
//		List<Double> smallestTrainingErrors = new ArrayList<Double>();
//		List<List<List<Integer>>> allSmallestAssignmentErrors = new ArrayList<List<List<Integer>>>();
//		List<List<List<Integer>>> allBestPredictedVoices = new ArrayList<List<List<Integer>>>();
//
////		trainingStartTime = AuxiliaryTool.getTimeStamp();
//
////		// Create, if necessary, groundTruthVoiceLabels (which is needed for ErrorCalculatorTab.calculateAssignmentErrors())
////		if (modellingApproach == CHORD_REGRESS) {
////			groundTruthVoiceLabels = new ArrayList<List<Double>>();
////			for (List<List<Double>> l : groundTruthChordVoiceLabels) {
////				groundTruthVoiceLabels.addAll(l);
////			}
////		}
//
//		// For each training run
//		int numberOfRuns = 1;
////		int numberOfRuns = argTrainingInfoMap.get(NUMBER_OF_RUNS).intValue();
//		for (int run = 0; run < numberOfRuns; run++) {  	
//			System.out.println("... Performing run " + run + " ...");
//
//			// 1. Determine bestWeightsFile and initialise the network weights
////			File bestWeightsFile = null;
////			File initialWeightsFile = null;
//			File bestWeightsFile = new File(path + bestWeightsFileName + ".xml");
////			File initialWeightsFile = new File(path + initWeightsFileName + ".xml");
////			if (learningApproach == NOTE_TO_NOTE) {
//////				bestWeightsFile = new File(prefix + "Best weights NOTE_TO_NOTE run " + run + ".xml");
////				bestWeightsFile = new File(path + bestWeightsFileNoteToNotePrefix + ".xml");
////				initialWeightsFile = new File(path + initialWeightsFileNoteToNotePrefix + ".xml");
//////				if (storeAdditionalFiles) {
/////					bestWeightsFile = new File(path + bestWeightsFileNoteToNotePrefix + " run " + run + ".xml");
//////					initialWeightsFile = new File(path + initialWeightsFileNoteToNotePrefix + " run " + run + ".xml");
//////				}
////			}
////			if (learningApproach == CHORD_TO_CHORD) {
//////				bestWeightsFile = new File(prefix + "Best weights CHORD_TO_CHORD run " + run + ".xml");
////				bestWeightsFile = new File(path + bestWeightsFileChordToChordPrefix + ".xml");
////				initialWeightsFile = new File(path + initialWeightsFileChordToChordPrefix + ".xml");
//////				if (storeAdditionalFiles) {
//////					bestWeightsFile = new File(path + bestWeightsFileChordToChordPrefix + " run " + run + ".xml");
//////					initialWeightsFile = new File(path + initialWeightsFileChordToChordPrefix + " run " + run + ".xml");
//////				}	
////			}
//			try {
//				if (!bestWeightsFile.exists()) {
//					bestWeightsFile.createNewFile();
//				}
//			} catch (IOException e) {
//				e.printStackTrace();
//			} 
//
////			List<Double> initialWeights = null;
////			File iniWeightsFile = new File(path + initWeightsFileName + ".xml");
////			initWeights(weightsInitialisationMethod);
////			initWeights(weightsInitialisationMethod, iniWeightsFile);
////			if (weightsInitinalisationMethod == RANDOM_INIT) {
////				initWeightsRandomly();
////				initialWeights = getNetworkWeights();
////				// Store the initial weights
////				AuxiliaryTool.storeObject(initialWeights, iniWeightsFile);
////			}
////			if (weightsInitialisationMethod == INIT_FROM_FILE) {
////				initialWeights = 
////					AuxiliaryTool.getStoredObject(new ArrayList<Double>(), iniWeightsFile);
////				initWeightsFromList(initialWeights);
////			}
//
////			// Initialise the network weights 
//			File iniWeightsFile = new File(path + initWeightsFileName + ".xml");
//			if (weightsInitialisationMethod == RANDOM_INIT) {
//				initWeights(null);
//				// Also store the initial weights
//				List<Double> iniWeights = getNetworkWeights();
//				ToolBox.storeObject(iniWeights, iniWeightsFile);
//			}
//			else if (weightsInitialisationMethod == INIT_FROM_LIST) {
//				List<Double> initialWeights = 
//					ToolBox.getStoredObject(new ArrayList<Double>(), iniWeightsFile);
//				System.out.println(iniWeightsFile);
//				initWeights(initialWeights);
//			}
//			
//			// 2. For each metaCycle:
//			double[] currentTrainingError = null; // NB: trainingError[1] is in both the NOTE_TO_NOTE and CHORD_TO_CHORD approach the error after training 
//			double currentRelativeError = -1.0; // CHORD_TO_CHORD only; remains -1.0 in the NOTE_TO_NOTE approach
//			double[] smallestTrainingError = null;
//			double smallestRelativeError = -1.0;
//			double currentClassificationError = -1.0;
//			double smallestClassificationError = 0.0;
//			List<List<Integer>> bestPredictedVoices = null;
//			List<List<Integer>> currentAssignmentErrors = null;
//			List<List<Integer>> smallestAssignmentErrors = null;
//			List<double[]> bestAllNetworkOutputs = null;
//			List<Double> bestAllHighestNetworkOutputs= null;
//			List<List<Integer>> bestAllBestVoiceAssignments = null;
//			List<Double> currentWeights = null;
//			List<Double> bestWeights = null;
//			String allClassErrs = "TRAINING ERRORS PER METACYCLE" + "\r\n"; 
//
//			int metaCycles = 0; 
//			do {
//				System.out.println("\nmetaCycle = " + metaCycles);
//				// 1. Train the network x times (where x is the number of cycles as defined in argTrainingInfoMap), and
//				if (modellingApproach == NOTE_CLASS) {
//					fixFlatSpot = true; // was false before 10-2-15
////					BasicMLDataSet trainingSet = 
////						createDataset(trainingData.get(0), trainingData.get(1));	
//					currentTrainingError = trainNetwork(modelParameters, fixFlatSpot, trainingSetN2N);			
//				} 
//				if (modellingApproach == CHORD_REGRESS) {
//					fixFlatSpot = true; // was false before 10-2-15
////					List<RelativeTrainingExample> argRelativeTrainingExamples = 
////						generateAllRelativeTrainingExamples(trainingData);	
//					currentTrainingError = trainNetworkRelative(modelParameters, 
//						fixFlatSpot, argRelativeTrainingExamples);
//					currentRelativeError = currentTrainingError[3]; 
//				}
//
//				// Set allNetworkOutputs
//				ArrayList<double[]> argAllNetworkOutputs = null;
//				List<Double> argAllHighestNetworkOutputs = null;
//				List<List<Integer>> argAllBestVoiceAssignments = null;
//				if (modellingApproach == NOTE_CLASS) {
////					setAllNetworkOutputs(); // ISNEW
////					allNetworkOutputs = createAllNetworkOutputs(noteFeatures); // NIW
////					allNetworkOutputs = createAllNetworkOutputs(trainingData.get(0));
////					setAllNetworkOutputs(createAllNetworkOutputs(trainingData.get(0)));
//					argAllNetworkOutputs = createAllNetworkOutputs(trainingData.get(0));
//				}
//				if (modellingApproach == CHORD_REGRESS) {
////					setAllHighestNetworkOutputs(); // ISNEW
////					allHighestNetworkOutputs = createAllHighestNetworkOutputs(chordFeatures); // NIW
////					allHighestNetworkOutputs = createAllHighestNetworkOutputs(trainingData); // NIW
////					setAllHighestNetworkOutputs(createAllHighestNetworkOutputs(trainingData));
//					argAllHighestNetworkOutputs	= createAllHighestNetworkOutputs(trainingData);
////					allBestVoiceAssignments = createAllBestVoiceAssignments(chordFeatures,
////						possibleVoiceAssignmentsAllChords);
////					allBestVoiceAssignments = createAllBestVoiceAssignments(trainingData,
////						possibleVoiceAssignmentsAllChords);
////					allBestVoiceAssignments = createAllBestVoiceAssignments(trainingData,
////						getPossibleVoiceAssignmentsAllChords());
////					argAllBestVoiceAssignments = createAllBestVoiceAssignments(trainingData,
////						getPossibleVoiceAssignmentsAllChords()); 6 maart
//					argAllBestVoiceAssignments = createAllBestVoiceAssignments(trainingData,
//						argPossibleVoiceAssignmentsAllChords);
//				}
//
//				// 2. Determine the assignmentErrors and the classificationError of the current
//				// trained network
////				List<List<Integer>> currentAllPredictedVoices = 
////					determinePredictedVoicesAndDurations(modelParameters).get(0);
////				List<List<List<Integer>>> predVoicesAndDur = 
////					outputEvaluator.determinePredictedVoicesAndDurations(modelParameters, 
////					allNetworkOutputs, allBestVoiceAssignments);
////				List<List<List<Integer>>> predVoicesAndDur = 
////					outputEvaluator.determinePredictedVoicesAndDurations(modelParameters, 
////					argAllNetworkOutputs, argAllBestVoiceAssignments);
////				List<List<Integer>> currentAllPredictedVoices = 
////					determinePredictedVoicesAndDurations(modelParameters, allNetworkOutputs).get(0);
////				List<List<Integer>> currentAllPredictedVoices = predVoicesAndDur.get(0);
//				List<List<Integer>> currentAllPredictedVoices = 
//					outputEvaluator.determinePredictedVoices(modelParameters, 
//					argAllNetworkOutputs, argAllBestVoiceAssignments);
//				
////				List<List<Double>> currentAllPredictedDurationLabels = null;
//				List<Rational[]> currentAllPredictedDurations = null;
//////			List<List<Double>> argGroundTruthDurationLabels = getGroundTruthDurationLabels();
////				if (groundTruthDurationLabels != null) {     
//				if (argGroundTruthDurationLabels != null) { 
////					currentAllPredictedDurationLabels = new ArrayList<List<Double>>();
////					currentAllPredictedDurations = new ArrayList<Rational[]>();
////					List<List<Integer>> currentAllPredictedDurationsAsInt =
////						determinePredictedVoicesAndDurations(modelParameters, allNetworkOutputs).get(1);
////					List<List<Integer>> currentAllPredictedDurationsAsInt = predVoicesAndDur.get(1);
////					for (List<Integer> l : currentAllPredictedDurationsAsInt) {
////						currentAllPredictedDurationLabels.add(dataConverter.convertIntoDurationLabel(l));
////						List<Double> currentDurationLabel = dataConverter.convertIntoDurationLabel(l);
////						currentAllPredictedDurations.add(dataConverter.convertIntoDuration(currentDurationLabel));
////					}
//					currentAllPredictedDurations = 
//						outputEvaluator.determinePredictedDurations(modelParameters, 
//						argAllNetworkOutputs, argAllBestVoiceAssignments); 
//				}
//
//				// When using bwd model:
//				// currentAllPredictedVoices = bwd 
//				// groundTruthVoiceLabels = bwd
//				// currentAllPredictedDurationLabels = bwd 
//				// groundTruthDurationLabels = bwd
//				// equalDurationUnisonsInfo = bwd
////				currentAssignmentErrors = errorCalculator.calculateAssignmentErrors(currentAllPredictedVoices, 
////					groundTruthVoiceLabels,	currentAllPredictedDurations, groundTruthDurationLabels,
////					equalDurationUnisonsInfo);
////				currentAssignmentErrors = errorCalculator.calculateAssignmentErrors(currentAllPredictedVoices, 
////					argGroundTruthVoiceLabels, currentAllPredictedDurations, groundTruthDurationLabels,
////					equalDurationUnisonsInfo);
////				currentAssignmentErrors = errorCalculator.calculateAssignmentErrors(currentAllPredictedVoices, 
////					argGroundTruthVoiceLabels, currentAllPredictedDurations, argGroundTruthDurationLabels,
////					equalDurationUnisonsInfo);
//				currentAssignmentErrors = errorCalculator.calculateAssignmentErrors(currentAllPredictedVoices, 
//					argGroundTruthVoiceLabels, currentAllPredictedDurations, argGroundTruthDurationLabels,
//					argEqualDurationUnisonsInfo);
//
////				currentClassificationError = 
////					errorCalculator.calculateClassificationError(currentAssignmentErrors, 
////						errorMeasurementIsStrict); 
//				currentClassificationError = 
//					errorCalculator.calculateClassificationError(currentAssignmentErrors); 
//
//				currentWeights = getNetworkWeights();
//				System.out.println("Network error: " + currentTrainingError[1]);
//				System.out.println("Classification error: " + currentClassificationError);
//				allClassErrs = allClassErrs.concat("" + currentClassificationError + "\r\n");
//
//				// 3. (Re)set the appropriate variables and store the network weights
//				// Determine which error to use as measure. In the NOTE_TO_NOTE approach, this should be the classification
//				// error; in the CHORD_TO_CHORD approach, it can also be the relative error
//				double currentErrorToTrack;
//				double smallestErrorSoFar;
//				if (modellingApproach == NOTE_CLASS) {
//					currentErrorToTrack = currentClassificationError;
//					smallestErrorSoFar = smallestClassificationError;
//				}
//				else {
//					currentErrorToTrack = currentClassificationError;
//					smallestErrorSoFar = smallestClassificationError;
////					currentErrorToTrack = currentRelativeError; 
////					smallestErrorSoFar = smallestRelativeError;
//				}
//				// a. Set and store if this is the first metaCycle
//				// b. Reset and store if 
//				// (1) the current error is smaller than the smallest error so far, or 
//				// (2) the current error is equal to the smallest error so far but the current
//				// network error is smaller
//				if (metaCycles == 0 || (currentErrorToTrack < smallestErrorSoFar) ||
//					(currentErrorToTrack == smallestErrorSoFar && (currentTrainingError[1] < smallestTrainingError[1]))) {     	
//					smallestClassificationError = currentClassificationError;
//					smallestTrainingError = currentTrainingError;
//					bestPredictedVoices = currentAllPredictedVoices;
//					smallestAssignmentErrors = currentAssignmentErrors;
//					bestWeights = currentWeights; 
////					AuxiliaryTool.storeObject(bestWeights, bestWeightsFile);
////					bestAllNetworkOutputs = allNetworkOutputs;
//					bestAllNetworkOutputs = argAllNetworkOutputs;
////					bestAllHighestNetworkOutputs = allHighestNetworkOutputs;
//					bestAllHighestNetworkOutputs = argAllHighestNetworkOutputs;
////					bestAllBestVoiceAssignments = allBestVoiceAssignments;
//					bestAllBestVoiceAssignments = argAllBestVoiceAssignments;
//					if (modellingApproach == CHORD_REGRESS) {
//						smallestRelativeError = currentRelativeError;
//					}	           
//				}
//				metaCycles++;
//			} while (metaCycles < maxMetaCycles); 
//
//			// 3. After each training run: add the final values of smallestClassificationError, smallestTrainingError,
//			// bestPredictedVoices, and smallestAssignmentsErrors to the Lists
//			smallestClassificationErrors.add(smallestClassificationError);
//			smallestTrainingErrors.add(smallestTrainingError[1]);
//			allBestPredictedVoices.add(bestPredictedVoices);
//			allSmallestAssignmentErrors.add(smallestAssignmentErrors);
////			allNetworkOutputs = bestAllNetworkOutputs;
//			setAllNetworkOutputs(bestAllNetworkOutputs);
////			allHighestNetworkOutputs = bestAllHighestNetworkOutputs;
//			setAllHighestNetworkOutputs(bestAllHighestNetworkOutputs);
////			allBestVoiceAssignments = bestAllBestVoiceAssignments;
//			setAllBestVoiceAssignments(bestAllBestVoiceAssignments);
//			// Set the network with the best weights (for calculating training results) 
//			// and also store them
//			initWeights(bestWeights);
//			ToolBox.storeObject(bestWeights, bestWeightsFile);
//
//			// 4. Store the training record of the current run
////			trainingEndTime = AuxiliaryTool.getTimeStamp();
////			int numTrainingEx = -1;
//			// NEW: calculate errorSpecifications here and give as arg to storeTrainingRecordCurrentRun()
////			int highestNumberOfVoicesTraining = modelParameters.get(HIGHEST_NUMBER_OF_VOICES).intValue();
////			boolean argModelDuration = false;
////			if (!isBidirectional) {
////				if (modelDuration) {
////					argModelDuration = true;
////				}
////			}
////			if (isBidirectional) {
////				if (modelDuration && modelDurationAgain) {
////					argModelDuration = true;
////				}
////			}
////			String errorSpecifications = 
////				errorCalculator.getErrorSpecifications(modelParameters, smallestAssignmentErrors, 
////				smallestTrainingError[1], bestPredictedVoices, groundTruthVoiceLabels,
////				equalDurationUnisonsInfo, /*highestNumberOfVoicesTraining,*/ true/*, argModelDuration*/);
////			String errorSpecifications = 
////				errorCalculator.getErrorSpecifications(modelParameters, smallestAssignmentErrors, 
////				smallestTrainingError[1], bestPredictedVoices, argGroundTruthVoiceLabels,
////				equalDurationUnisonsInfo, true);
//			
//			trainingResults.add(smallestAssignmentErrors);
//			trainingResults.add(bestPredictedVoices);
//			
//			setSmallestNetworkError(smallestTrainingError[1]);
//			setAllClassificationErrors(allClassErrs);
//			
////			String errorSpecifications = 
////					errorCalculator.getErrorSpecifications(modelParameters, smallestAssignmentErrors, 
////					smallestTrainingError[1], bestPredictedVoices, argGroundTruthVoiceLabels,
////					argEqualDurationUnisonsInfo, true);
//			
////			getTrainingResults(modelParameters,  /*smallestAssignmentErrors, 
////				bestPredictedVoices,*/ path, /*bestWeights,*/ allClassErrs, errorSpecifications,
////				trainingSettings/*, argVoicesCoDNotes*/);
////			if (storeAdditionalFiles) {
////				AuxiliaryTool.storeTextFile(allClassErrs, new File(path + "Training errors.txt"));
////			}
//		}
//		return trainingResults;
//
////		// After the last training run: store the training results
////		if (storeAdditionalFiles) {
////			trainingEndTime = AuxiliaryTool.getTimeStamp();
////			storeTrainingResults(argTrainingInfoMap, smallestClassificationErrors, allSmallestAssignmentErrors,
////				allBestPredictedVoices, smallestTrainingErrors, path);
////		}
		return null;
	}


//	public void trainMultipleRuns(int learningApproach, int epoch, OutputEvaluator outputEvaluator, 
//			ErrorCalculator errorCalculator, DataConverter dataConverter, String prefix) {
//	}


	/**
	 * Trains the network (using relative training), using the settings as stored in the map.
	 * 
	 * @param map
	 * @param fixFlatSpot
	 * @return An Array with 4 doubles: initial Network error (SSE), final Network error (SSE),
	 *         initial relative error (classification), final relative error (classification)
	 */
	public double[] trainNetworkRelative(Map<String, Double> map, boolean fixFlatSpot, 
		List<RelativeTrainingExample> argRelativeTrainingExamples) {
		double cyclesAsDouble = map.get(CYCLES);
		int cycles = (int) cyclesAsDouble;
		double alpha = map.get(LEARNING_RATE);
		double lambda = map.get(REGULARISATION_PARAMETER);
		double epsilon = map.get(MARGIN);

		System.out.println("============Start training============");
		System.out.println("Starting the relative training with " + argRelativeTrainingExamples.size() + " pairs of training examples " + 
			"for max " + cycles + " cycles.");

		double finalError = -1;
		double firstError = -1;

		BasicMLDataSet trainingSet = new BasicMLDataSet();
		// This MLDataPair is necessary because the trainingSet cannot be empty when creating a BackPropagation object 
		MLDataPair dataPair = BasicMLDataPair.createPair(argRelativeTrainingExamples.get(0).getBetterVal().size(), 1);
		trainingSet.add(dataPair);
		// Backpropagation train = new Backpropagation(network, trainingSet);
		// train.setLearningRate(alpha);	
		ResilientPropagation train = new ResilientPropagation(network, trainingSet);
		train.setRPROPType(RPROPType.iRPROPp); // was classic RPROP before 10-2-15
		train.fixFlatSpot(fixFlatSpot); 
		train.setThreadCount(1);

		double[] error = new double[4];
		int numOfNotSatisfiedTrainingExamples = 0;
		double firstRelClassError = -1;
		// For each cycle
		for (int i = 0; i < cycles; i++) {
			List<MLDataPair > trainData = new ArrayList<MLDataPair>();
			numOfNotSatisfiedTrainingExamples = 0;		

			// For each rte: check whether it satisfies the condition (the ground truth feature vector should get a 
			// higher network output than the other feature vector). If not: create MLDataPairs from them and add 
			// them to the training data
			for (RelativeTrainingExample rte : argRelativeTrainingExamples) {
				List<Double> groundTruthFeatureVector = rte.getBetterVal();
				List<Double> otherFeatureVector = rte.getWorseVal();
				// Get the network output for the ground truth feature vector (GTFV) and that for the other feature
				// vector (OFV)
//				double[] outputGroundTruthFeatureVector = evalNetwork(rte.getBetterVal()); // was "outputBetter"
				double[] outputGroundTruthFeatureVector = evalNetwork(groundTruthFeatureVector);
//				double[] outputOtherFeatureVector = evalNetwork(rte.getWorseVal()); // was "outputWorse"
				double[] outputOtherFeatureVector = evalNetwork(otherFeatureVector); // was "outputWorse"

				// Check whether outputGTFV is higher
				switch (COMPARATOR) {
					// a. Sigmoid  
					case SGM:
						double compOut = 
							1 / (1 + Math.exp(gamma * (outputOtherFeatureVector[0] - outputGroundTruthFeatureVector[0])));
						// Then switch the output values for the training
						double compDeriv = compOut * (1 - compOut);

						if (compOut > 0.45) {
							numOfNotSatisfiedTrainingExamples++;
							MLDataPair better = BasicMLDataPair.createPair(groundTruthFeatureVector.size(), 1);
							MLDataPair worse = BasicMLDataPair.createPair(otherFeatureVector.size(), 1);
							double[] inputsBetter = new double[groundTruthFeatureVector.size()];
							double[] inputsWorse = new double[otherFeatureVector.size()];
							// Convert the List<Double> into double[] while switching
							for (int j = 0; j < groundTruthFeatureVector.size(); j++) {
								inputsBetter[j] = groundTruthFeatureVector.get(j);
								inputsWorse[j] = otherFeatureVector.get(j);
							}
							// double outpAvg = (outputBetter[0] + outputWorse[0])/2;
							// Fill better training example
							better.setInputArray(inputsBetter);
							double[] btarg = new double[] { outputGroundTruthFeatureVector[0] + (compOut * compDeriv * gamma) };
							better.setIdealArray(btarg);
							trainData.add(better);
							// Fill worse training example
							worse.setInputArray(inputsWorse);
							double[] wtarg = new double[] { outputOtherFeatureVector[0] - (compOut * compDeriv * gamma) };
							worse.setIdealArray(wtarg);
							trainData.add(worse);
						}
						break;
					// b. Average  
					case AVG:
						if (outputOtherFeatureVector[0] > outputGroundTruthFeatureVector[0] - epsilon) {
							// Switch the output values for the training
							MLDataPair better = BasicMLDataPair.createPair(groundTruthFeatureVector.size(), 1);
							MLDataPair worse = BasicMLDataPair.createPair(otherFeatureVector.size(), 1);
							double[] inputsBetter = new double[groundTruthFeatureVector.size()];
							double[] inputsWorse = new double[otherFeatureVector.size()];
							// Convert the List<Double> into double[] while switching
							for (int j = 0; j < groundTruthFeatureVector.size(); j++) {
								inputsBetter[j] = groundTruthFeatureVector.get(j);
								inputsWorse[j] = otherFeatureVector.get(j);
							}
							double outpAvg = (outputGroundTruthFeatureVector[0] + outputOtherFeatureVector[0]) / 2;
							outputOtherFeatureVector[0] = outpAvg + epsilon;
							outputGroundTruthFeatureVector[0] = outpAvg - epsilon;
							// Construction of the MLDataPairs; switching happens here:
							better.setInputArray(inputsBetter);
							better.setIdealArray(outputOtherFeatureVector);
							worse.setInputArray(inputsWorse);
							worse.setIdealArray(outputGroundTruthFeatureVector);
							trainData.add(better);
							trainData.add(worse);
							numOfNotSatisfiedTrainingExamples++;
						}
						break;
					// c. Semilinear
					case SLN:
						// If outputGTFV is not higher than outputOFV by at least the given margin epsilon (i.e., if it is not
						// at least epsilon higher than outputOFV): rte does not satisfy condition; further training necessary
						if (outputOtherFeatureVector[0] > outputGroundTruthFeatureVector[0] - epsilon) {
							numOfNotSatisfiedTrainingExamples++;		
							// Make BasicMLDataPairs better and worse, which will contain an input that is a feature vector and
							// an output that is a rating (a network output) 
							// a. Create empty pairs 
							MLDataPair better = BasicMLDataPair.createPair(groundTruthFeatureVector.size(), 1);
							MLDataPair worse = BasicMLDataPair.createPair(otherFeatureVector.size(), 1);
							// b. Create inputs: the GTFV and the OFV, respectively, both as a double[] 
							double[] inputsBetter = new double[groundTruthFeatureVector.size()];
							double[] inputsWorse = new double[otherFeatureVector.size()];
							for (int j = 0; j < groundTruthFeatureVector.size(); j++) {
								inputsBetter[j] = groundTruthFeatureVector.get(j);
								inputsWorse[j] = otherFeatureVector.get(j);
							}

							// c. Modify outputs
							outputOtherFeatureVector[0] += epsilon;
							outputGroundTruthFeatureVector[0] -= epsilon;
							// d. Fill the BasicMLDataPairs, thereby switching the current network outputs. Because we want the 
							// ground truth feature vector to result in a higher network output than the other feature vector, we
							// give the MLDataPair better the higher output (i.e., the one the network currently gives for the 
							// other feature vector), and the MLDataPair worse the lower output (i.e., the one the network 
							// currently gives for the ground truth feature vector)
							better.setInputArray(inputsBetter);
							better.setIdealArray(outputOtherFeatureVector);
							worse.setInputArray(inputsWorse);
							worse.setIdealArray(outputGroundTruthFeatureVector);
							// Add the MLDataPairs to the training data
							trainData.add(better);
							trainData.add(worse);
						}
						// TODO added 17-11
//						else {
//					  	// a. Create empty pairs 
//					  	MLDataPair better = BasicMLDataPair.createPair(groundTruthFeatureVector.size(), 1);
//						  MLDataPair worse = BasicMLDataPair.createPair(otherFeatureVector.size(), 1);
//						  // b. Create inputs: the GTFV and the OFV, respectively, both as a double[] 
//						  double[] inputsBetter = new double[groundTruthFeatureVector.size()];
//						  double[] inputsWorse = new double[otherFeatureVector.size()];
//						  for (int j = 0; j < groundTruthFeatureVector.size(); j++) {
//							  inputsBetter[j] = groundTruthFeatureVector.get(j);
//							  inputsWorse[j] = otherFeatureVector.get(j);
//						  }
//					    
//					    // c. Fill the BasicMLDataPairs
//					    better.setInputArray(inputsBetter);
//					    better.setIdealArray(outputGroundTruthFeatureVector);
//					    worse.setInputArray(inputsWorse);
//					    worse.setIdealArray(outputOtherFeatureVector);
//					    // Add the MLDataPairs to the training data
//					    trainData.add(better);
//					    trainData.add(worse);
//					  }
				}
			}	// All relativeTrainingExamples dealt with 
			if (DEBUG) {
				System.out.println("Rel Class Errs after " + (i + 1) + " cycles: " + numOfNotSatisfiedTrainingExamples);
			}
//			System.out.println("  size trainData = " + trainData.size());
			// If numOfNotSatisfiedTrainingExamples == 0, i.e., if all values are assigned correctly, the training can 
			// be stopped
			if (numOfNotSatisfiedTrainingExamples == 0) { 
				finalError = train.getError();
				break; // from outer for
			}
			// If not: train the network (batch training)
			trainingSet.setData(trainData);
			train.setTraining(trainingSet);
			train.iteration(1);
			// If the current cycles loop is the first, i.e., if firstError has not been set before: get the initial error
			if (firstError == -1) {
				firstError = train.getError();
				firstRelClassError = numOfNotSatisfiedTrainingExamples;
			}
			// Regularise
			regularise(alpha * lambda );
			if (DEBUG) {
				System.out.println("Error after " + (i + 1) + " cycles = " + train.getError());
//			// when doing last cycle
//			if (i == cycles - 1) {
//				finalError = train.getError();
//				storeWeights(bestWeightsFile);
//			}
			}
		} // end cycles loop
		finalError = train.getError();
		double finalRelClassError = numOfNotSatisfiedTrainingExamples;
		if (DEBUG) {
			System.out.println("============End training============ \n");
			System.out.println("Error before training = " + firstError + " \n");
			System.out.println("Error after training = " + finalError);
		}
		error[0] = firstError;
		error[1] = finalError;
		error[2] = firstRelClassError / argRelativeTrainingExamples.size();
		error[3] = finalRelClassError / argRelativeTrainingExamples.size();
		System.out.println("---> firstRelClassError = " + firstRelClassError);
		System.out.println("---> finalRelClassError = " + finalRelClassError);

		return error;
	}


	/**
	 * Evaluates the network by calculating the output (label) for the given input (features).
	 * 
	 * @param argFeatures
	 * @return
	 */
	public double[] evalNetwork(List<Double> argFeatures) {
		double[] featuresAsArray = new double[argFeatures.size()];
		for (int i = 0; i < featuresAsArray.length; i++) {
			featuresAsArray[i] = argFeatures.get(i);	
		}	
		double[] output = new double[network.getOutputCount()]; 
		network.compute(featuresAsArray, output);

		return output;
	}


	ArrayList<double[]> createAllNetworkOutputs(List<List<Double>> argNoteFeatures) {
		ArrayList<double[]> allNetwOutp = new ArrayList<double[]>();
		for (int i = 0; i < argNoteFeatures.size(); i++) {
			double[] predictedLabel = evalNetwork(argNoteFeatures.get(i));
			allNetwOutp.add(predictedLabel);
		}
		return allNetwOutp;
	}


	private List<Double> createAllNetworkOutputsForChord(List<List<Double>> currentChordFeatures) {
		List<Double> currentNetworkOutputs = new ArrayList<Double>();
		for (int j = 0; j < currentChordFeatures.size(); j++) {
			List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
			double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
			currentNetworkOutputs.add(currentNetworkOutput[0]);
			if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
				System.out.println("Network output is NaN.");
				System.exit(0);
			}
		}
		return currentNetworkOutputs;
	}

	
	/**
	 * Creates the list of network outputs, one for each chord feature vector, per chord.
	 * 
	 * @param argChordFeatures
	 * @return
	 */
	public List<List<Double>> createAllNetworkOutputsForAllChords(List<List<List<Double>>> 
		argChordFeatures) {
		List<List<Double>> all = new ArrayList<List<Double>>();
		int numberOfChords = argChordFeatures.size();
		// For each chord
		for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
			List<List<Double>> currentChordFeatures = argChordFeatures.get(chordIndex);
//			all.add(createAllNetworkOutputsForChord(currentChordFeatures));
			List<Double> currentNetworkOutputs = new ArrayList<Double>();
			// For each chord feature
			for (int j = 0; j < currentChordFeatures.size(); j++) {
				List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
				double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
				currentNetworkOutputs.add(currentNetworkOutput[0]);
				if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
					System.out.println("Network output is NaN.");
					System.exit(0);
				}
			}
			all.add(currentNetworkOutputs);
		}
		return all;
	}


	private List<Double> createAllHighestNetworkOutputs(List<List<Double>> 
		allNetworkOutputsForAllChords) {		
		// allHighestNetworkOutputs, as well as allBestVoiceAssignments, must be recreated every time 
		// this method is called in the training case. In the test case, this method is only called once, so
		// recreation is not strictly necessary -- but it is harmless nevertheless
		List<Double> allHiNetwOutp = new ArrayList<Double>();
		// For each chord
		for (List<Double> currentNetworkOutputs : allNetworkOutputsForAllChords) {
//		int numberOfChords = argChordFeatures.size();
//		for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
			// a. For all possible feature vectors for this chord: evaluate the network and
			// add the result, the network output, to currentNetworkOutputs
//			List<List<Double>> currentChordFeatures = argChordFeatures.get(chordIndex);

//			List<Double> currentNetworkOutputs = 
//				createAllNetworkOutputsForChord(currentChordFeatures);
//			List<Double> currentNetworkOutputs = 
//				allNetworkOutputsForAllChords.get(chordIndex);

			// b. Add the highest network output to the list; it does not matter whether 
			// it appears more than once (if this happens, it is solved within
			// determineBestVoiceAssignment(), which is called in training, test, and
			// application case)
			double currentHighestNetworkOutput = Collections.max(currentNetworkOutputs);
			allHiNetwOutp.add(currentHighestNetworkOutput);
		}	
		return allHiNetwOutp;
	}

	// FIXME commented out 12.11.2016
	private List<List<Integer>> createAllBestVoiceAssignments(
		List<List<Double>> allNetworkOutputsForAllChords, 
		List<List<List<Integer>>> argPossibleVoiceAssignmentsAllChords) { 
//		
//		// allBestVoiceAssignments must be recreated every time this method is called 
//		// in the training case. In the test case, this method is only called once, so
//		// recreation is not strictly necessary -- but it is harmless nevertheless
//		List<List<Integer>> allBestVoiceAss = new ArrayList<List<Integer>>();
//		// For each chord
////		int numberOfChords = argChordFeatures.size();
//		int numberOfChords = allNetworkOutputsForAllChords.size();
//		for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
//			// a. For all possible feature vectors for this chord: evaluate the network and
//			// add the result, the network output, to currentNetworkOutputs
////			List<List<Double>> currentChordFeatures = argChordFeatures.get(chordIndex);
//			List<List<Integer>> currentPossibleVoiceAssignments = 
//				argPossibleVoiceAssignmentsAllChords.get(chordIndex);  	
//			
////			List<Double> currentNetworkOutputs =
////				createAllNetworkOutputsForChord(currentChordFeatures);
//			List<Double> currentNetworkOutputs =
//				allNetworkOutputsForAllChords.get(chordIndex);
//
//			// b. Determine the best voice assignment and add it to the list. Because it is
//			// possible that different voice assignments result in the same network output,
//			// the highest output may occur multiple times. If this is the case, less likely
//			// candidates must be filtered out; this happens inside determineBestVoiceAssignment(). TODO?
//			List<Integer> predictedBestVoiceAssignment = 
//				outputEvaluator.determineBestVoiceAssignment(currentNetworkOutputs, 
//					currentPossibleVoiceAssignments); 
//			allBestVoiceAss.add(predictedBestVoiceAssignment);
//		}
//		return allBestVoiceAss;
		return null;
	}


	private List<Double> createAllHighestNetworkOutputsOLD(List<List<List<Double>>> argChordFeatures) {
		// allHighestNetworkOutputs, as well as allBestVoiceAssignments, must be recreated every time 
		// this method is called in the training case. In the test case, this method is only called once, so
		// recreation is not strictly necessary -- but it is harmless nevertheless
//		allHighestNetworkOutputs = new ArrayList<Double>();
		List<Double> allHiNetwOutp = new ArrayList<Double>();
//		allBestVoiceAssignments = new ArrayList<List<Integer>>();
		// For each chord
		int numberOfChords = argChordFeatures.size();
		for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
			// a. For all possible feature vectors for this chord: evaluate the network and
			// add the result, the network output, to currentNetworkOutputs
			List<List<Double>> currentChordFeatures = argChordFeatures.get(chordIndex);
//			List<List<Integer>> currentPossibleVoiceAssignments = 
//				possibleVoiceAssignmentsAllChords.get(chordIndex);  	
		
			List<Double> currentNetworkOutputs = 
				createAllNetworkOutputsForChord(currentChordFeatures);
//			// TURNED INTO METHOD createNetworkOutputsChord() -->
//			List<Double> currentNetworkOutputs = new ArrayList<Double>();
//			for (int j = 0; j < currentChordFeatures.size(); j++) {
//				List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
//				double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
//				currentNetworkOutputs.add(currentNetworkOutput[0]);
//				if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
//					System.out.println("Network output is NaN.");
//					System.exit(0);
//				}
//			}
//			// TURN INTO METHOD <--
			
			// b. Add the highest network output to the list; it does not matter whether 
			// it appears more than once (if this happens, it is solved within
			// determineBestVoiceAssignment(), which is called in training, test, and
			// application case)
			double currentHighestNetworkOutput = Collections.max(currentNetworkOutputs);
//			if (Collections.frequency(currentNetworkOutputs, currentHighestNetworkOutput) > 1) {
//			 	System.out.println("Highest network output appears more than once.");
//				System.exit(0);
//			}
//			allHighestNetworkOutputs.add(currentHighestNetworkOutput);
			allHiNetwOutp.add(currentHighestNetworkOutput);

//			// c. Determine the best voice assignment and add it to allBestVoiceAssignments. Because it is possible 
//			// that different voice assignments result in the same network output, the highest output may occur
//			// multiple times. If this is the case, less likely candidates must be filtered out; this happens 
//			// inside determineBestVoiceAssignment()
//			List<Integer> predictedBestVoiceAssignment = 
//				outputEvaluator.determineBestVoiceAssignment(currentBasicTabSymbolPropertiesChord, currentNetworkOutputs,
//				currentPossibleVoiceAssignments, isTrainingOrTestMode);
//				outputEvaluator.determineBestVoiceAssignment(currentNetworkOutputs, currentPossibleVoiceAssignments); 
//			allBestVoiceAssignments.add(predictedBestVoiceAssignment);
		}	
		return allHiNetwOutp;
	}

	// FIXME commented out 12.11.2016
	private List<List<Integer>> createAllBestVoiceAssignmentsOLD(List<List<List<Double>>> argChordFeatures,
		List<List<List<Integer>>> argPossibleVoiceAssignmentsAllChords) { 
//		
//		// allBestVoiceAssignments, must be recreated every time this method is called 
//		// in the training case. In the test case, this method is only called once, so
//		// recreation is not strictly necessary -- but it is harmless nevertheless
////		allBestVoiceAssignments = new ArrayList<List<Integer>>();
//		List<List<Integer>> allBestVoiceAss = new ArrayList<List<Integer>>();
//		// For each chord
//		int numberOfChords = argChordFeatures.size();
//		for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
//			// a. For all possible feature vectors for this chord: evaluate the network and
//			// add the result, the network output, to currentNetworkOutputs
//			List<List<Double>> currentChordFeatures = argChordFeatures.get(chordIndex);
//			List<List<Integer>> currentPossibleVoiceAssignments = 
//				argPossibleVoiceAssignmentsAllChords.get(chordIndex);  	
//			
//			List<Double> currentNetworkOutputs =
//				createAllNetworkOutputsForChord(currentChordFeatures);
////			// TURNED INTO METHOD createNetworkOutputsChord() -->
////			List<Double> currentNetworkOutputs = new ArrayList<Double>();
////			for (int j = 0; j < currentChordFeatures.size(); j++) {
////				List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
////				double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
////				currentNetworkOutputs.add(currentNetworkOutput[0]);
////				if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
////					System.out.println("Network output is NaN.");
////					System.exit(0);
////				}
////			}
////			// TURNED INTO METHOD <--
//			
////			// b. Add the highest network output to allHighestNetworkOutputs; it does not 
////			// matter whether it appears more than once (if this happens, it is solved 
////			// within determineBestVoiceAssignment(), which is called in training, test,
////			// and application case)
////			double currentHighestNetworkOutput = Collections.max(currentNetworkOutputs);
////			if (Collections.frequency(currentNetworkOutputs, currentHighestNetworkOutput) > 1) {
////			 	System.out.println("Highest network output appears more than once.");
////				System.exit(0);
////			}
////			allHighestNetworkOutputs.add(currentHighestNetworkOutput);
//
//			// b. Determine the best voice assignment and add it to the list. Because it is
//			// possible that different voice assignments result in the same network output,
//			// the highest output may occur multiple times. If this is the case, less likely
//			// candidates must be filtered out; this happens inside determineBestVoiceAssignment(). TODO?
//			List<Integer> predictedBestVoiceAssignment = 
////				outputEvaluator.determineBestVoiceAssignment(currentBasicTabSymbolPropertiesChord, currentNetworkOutputs,
////				currentPossibleVoiceAssignments, isTrainingOrTestMode);
//				outputEvaluator.determineBestVoiceAssignment(currentNetworkOutputs, 
//					currentPossibleVoiceAssignments); 
////			allBestVoiceAssignments.add(predictedBestVoiceAssignment);
//			allBestVoiceAss.add(predictedBestVoiceAssignment);
//		}
//		return allBestVoiceAss;
		return null;
	}


//	/**
//	 * Initialises the network weights.
//	 */
//	private void initWeights(int initMethod) {  	
//		if (initMethod == RANDOM_INIT) {
//			initWeightsRandomly();
//		}
////		if (initMethod == INIT_FROM_LIST) {
////			initWeightsFromList(weightsList);
////		}
//		if (initMethod == INIT_FROM_FILE) {
//			initWeightsFromFile(weightsFile);
//		}
//	}


//	public static void main(String[] args) {
//		// Make static: network / getNetworkWeights() / initWeightsRandomly() / reduceWeights()
//		
//		EncogNNManager e = new EncogNNManager(2, 2, 1);
//		System.out.println("layers = " + network.getLayerCount());
//		System.out.println("neurons L0 = " + network.getLayerNeuronCount(0));
//		System.out.println("neurons L0 = " + network.getLayerTotalNeuronCount(0));
//		System.out.println("neurons L1 = " + network.getLayerNeuronCount(1));
//		System.out.println("neurons L1 = " + network.getLayerTotalNeuronCount(1));
//		System.out.println("neurons L2 = " + network.getLayerNeuronCount(2));
//		System.out.println("neurons L2 = " + network.getLayerTotalNeuronCount(2));
//		 
//		
////		initWeightsRandomly();
////		System.out.println(getNetworkWeights().size()); 
////		System.out.println(getNetworkWeights()); 
////		regularise(0.001);
//		
//	}


	public void initWeights(List<Double> arg) {
//		if (arg == null) {
//			initWeightsRandomly();
//		}
//		else {
//			initWeightsFromList(arg);
//		}
		setNetworkWeights(arg);
	}


	/**
	 * Sets all network weights. In case <code> arg == null </code>, the weights are set
	 * randomly; otherwise they are set from the list.
	 * 
	 * @param arg
	 */
	// TODO test
	private void setNetworkWeights(List<Double> arg) {
		int counter = 0;
		for (int fromLayer = 0; fromLayer < network.getLayerCount() - 1; fromLayer++) {
			for (int toNeuron = 0; toNeuron < network.getLayerNeuronCount(fromLayer + 1); toNeuron++) {
				for (int fromNeuron = 0; fromNeuron < network.getLayerTotalNeuronCount(fromLayer); fromNeuron++) { 
					double weight;
					// In case of random initialisation
					if (arg == null) {
						weight = (Math.random() - 0.5)/10;
					}
					// In case of initialisation from a list
					else {
						weight = arg.get(counter);
						counter++;
					}
					network.setWeight(fromLayer, fromNeuron, toNeuron, weight);
				}
			}
		}
	}
	
	
	/**
	 * Decrease all weights, except those coming from bias neurons, by a given factor. 
	 * The formula is <code>newWeight = oldWeight * (1 - factor)</code>
	 * 
	 * @param factor 
	 */
	// TODO test
	private void regularise(double factor) {
		// Regularise every single weight, but not the bias weights 
		for (int fromLayer = 0; fromLayer < network.getLayerCount() - 1; fromLayer++) {
//			System.out.println("fromL = " + fromLayer);
			for (int toNeuron = 0; toNeuron < network.getLayerNeuronCount(fromLayer + 1); toNeuron++) {
//				System.out.println("toN = " + toNeuron);
				// Regularise, for each neuron in the current fromLayer, the weight between that neuron and the 
				// current toNeuron in the next layer
				// Example for three-layer NN with 30 input, 10 hidden, and 5 output neurons 
				// 1. fromLayer = input (next layer = hidden)
				//    --> regularise input 0-29 to hidden 0, input 0-29 to hidden 1, ..., input 0-29 to hidden 9
				// 2. fromLayer = hidden (next layer = output)
				//    --> regularise hidden 0-9 to output 0, hidden 0-9 to output 1, ..., hidden 0-9 to output 4
				for (int fromNeuron = 0; fromNeuron < network.getLayerNeuronCount(fromLayer); fromNeuron++) {  
//					System.out.println("fromN = " + fromNeuron);
					// Get the unregularised weight, then regularise, then set the regularised weight  
					double currentWeight = network.getWeight(fromLayer, fromNeuron, toNeuron);
					currentWeight = currentWeight * (1 - factor);   
					network.setWeight(fromLayer, fromNeuron, toNeuron, currentWeight);
				}
			}
		} // end of regularisation
	}


    /**
     * Gets all network weights.
     * 
     * @return
     */
	// TODO test
	List<Double> getNetworkWeights() {
		List<Double> weights = new ArrayList<Double>(); 
		for (int fromLayer = 0; fromLayer < network.getLayerCount() - 1; fromLayer++) {
			for (int toNeuron = 0; toNeuron < network.getLayerNeuronCount(fromLayer + 1); toNeuron++) {
				for (int fromNeuron = 0; fromNeuron < network.getLayerTotalNeuronCount(fromLayer); fromNeuron++) {
					double weight = network.getWeight(fromLayer, fromNeuron, toNeuron);			
					weights.add(weight);
				}
			}
		}
		return weights;
	}


	// BELOW THIS LINE: methods not in use
	void initWeightsRandomly() {
		for (int fromLayer = 0; fromLayer < network.getLayerCount() - 1; fromLayer++) {
//			System.out.println("fromL = " + fromLayer);
			for (int toNeuron = 0; toNeuron < network.getLayerNeuronCount(fromLayer + 1); toNeuron++) {
//				System.out.println("toN = " + toNeuron);
				for (int fromNeuron = 0; fromNeuron < network.getLayerTotalNeuronCount(fromLayer); fromNeuron++) { 
//					System.out.println("fromN = " + fromNeuron);
					double weight = (Math.random() - 0.5)/10;
					network.setWeight(fromLayer, fromNeuron, toNeuron, weight);
				}
			}
		}
	}


	private void initWeightsFromList(List<Double> weightList) {
		int counter = 0;
		for (int fromLayer = 0; fromLayer < network.getLayerCount() - 1; fromLayer++) {
			for (int toNeuron = 0; toNeuron < network.getLayerNeuronCount(fromLayer + 1); toNeuron++) {
				for (int fromNeuron = 0; fromNeuron < network.getLayerTotalNeuronCount(fromLayer); fromNeuron++) { 
					double weight = weightList.get(counter);
					network.setWeight(fromLayer, fromNeuron, toNeuron, weight);
					counter++;
				}
			}
		}
	}


	private List<Double> accessWeights(boolean returnWeights, List<Double> argList, double factor) {
		List<Double> weights = new ArrayList<Double>();
		for (int fromLayer = 0; fromLayer < network.getLayerCount() - 1; fromLayer++) {
			for (int toNeuron = 0; toNeuron < network.getLayerNeuronCount(fromLayer + 1); toNeuron++) {
				int neuronCount = network.getLayerTotalNeuronCount(fromLayer);
				// If regularisation, neuroncount is different
				if (factor != -1) {
					neuronCount = network.getLayerNeuronCount(fromLayer);
				}
				for (int fromNeuron = 0; fromNeuron < neuronCount; fromNeuron++) {
					// Random initialisation
					if (returnWeights == false && argList == null) {
						
					}
					// Initialisation from list
					if (returnWeights == false && argList != null) {
						
					}
					// Regularisation
					if (returnWeights == false && factor != -1) {
						
					}
					if (returnWeights == true) {
						
					}
				}
			}
		}
		return weights;
	}


//	private void initWeightsFromFile(File file) {
//		List<Double> weightList = AuxiliaryTool.getStoredObject(new ArrayList<Double>(), file);
//		initWeightsFromList(weightList);
//	}


//	public void setWeightsList(List<Double> argWeightsList) {
//		weightsList = argWeightsList;
//	}


//	public void setWeightsFile(File argWeightsFile) {
//		weightsFile = argWeightsFile;
//	}


//	private void setAllNetworkOutputs() {
//   	// allNetworkOutputs must be recreated every time this method is called in the training 
//		// case. In the test case, this method is only called once, so recreation is not strictly 
//		// necessary -- but it is harmless nevertheless
//		allNetworkOutputs = new ArrayList<double[]>();
//		for (int i = 0; i < noteFeatures.size(); i++) {
//			double[] predictedLabel = evalNetwork(noteFeatures.get(i));
//			allNetworkOutputs.add(predictedLabel);
//		}
//	}


//	/**
//	 * Gets the voices and durations the network predicts for noteFeatures or chordFeatures with the given
//	 * settings. Returns a List<List<List<Integer>>> containing:
//	 *   as element 0: all the predicted voices 
//	 *   as element 1: all the predicted durations (as ints indicating their index + 1 in the durationLabel)
//	 * 
//	 * @param argTrainingInfoMap
//	 * @return
//	 */ 
//	public List<List<List<Integer>>> determinePredictedVoicesAndDurationsZOALSHETWAS(Map<String, Double> argTrainingInfoMap) { 
//		
//		List<List<List<Integer>>> allPredictedVoicesAndDurations = new ArrayList<List<List<Integer>>>();
//		List<List<Integer>> allPredictedVoices = new ArrayList<List<Integer>>();
//		List<List<Integer>> allPredictedDurations = new ArrayList<List<Integer>>();
//
//		int modellingApproach = argTrainingInfoMap.get(MODELLING_APPROACH).intValue();
//	    double allowCoD = argTrainingInfoMap.get(ALLOW_COD);
//	    double deviationThreshold = argTrainingInfoMap.get(DEVIATION_THRESHOLD);
//
//	    if (modellingApproach == NOTE_CLASS) { 
//	    	// allNetworkOutputs must be recreated every time this method is called in the training case. In the test
//	    	// case, this method is only called once, so recreation is not strictly necessary -- but it is harmless nevertheless
//	    	allNetworkOutputs = new ArrayList<double[]>();
//
//	    	for (int i = 0; i < noteFeatures.size(); i++) {
//	    		// Evaluate the network for the current onset
//				double[] predictedLabel = evalNetwork(noteFeatures.get(i));
//				allNetworkOutputs.add(predictedLabel);
//				// Interpret the output
//				// a. Get the predicted voice(s) and add them to allPredictedVoices 
////				List<Integer> predictedVoices = outputEvaluator.interpretNetworkOutput(predictedLabel, allowCoD, 
////					deviationThreshold).get(0);
//				allPredictedVoices.add(outputEvaluator.interpretNetworkOutput(predictedLabel, allowCoD, 
//				  deviationThreshold).get(0));
//				// b. If applicable: get the predicted duration(s) and add them to allPredictedDurations
//				if (predictedLabel.length > MAX_NUM_VOICES_N2N) {
//					allPredictedDurations.add(outputEvaluator.interpretNetworkOutput(predictedLabel, allowCoD, 
//						deviationThreshold).get(1));
//				}
//			}
//	    }
//	    if (modellingApproach == CHORD_REGRESS) {
//	    	int numberOfChords = chordFeatures.size();
//			// allHighestNetworkOutputs, as well as allBestVoiceAssignments, must be recreated every time 
//			// this method is called in the training case. In the test case, this method is only called once, so
//			// recreation is not strictly necessary -- but it is harmless nevertheless
//			allHighestNetworkOutputs = new ArrayList<Double>(); 
//			allBestVoiceAssignments = new ArrayList<List<Integer>>();
//			// For each chord
//			for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
//				// a. For all possible feature vectors for this chord: evaluate the network and add the result, the
//				// network output, to currentNetworkOutputs
//				List<List<Double>> currentChordFeatures = chordFeatures.get(chordIndex);
//				List<List<Integer>> currentPossibleVoiceAssignments = possibleVoiceAssignmentsAllChords.get(chordIndex);  	
//				List<Double> currentNetworkOutputs = new ArrayList<Double>();
//				for (int j = 0; j < currentChordFeatures.size(); j++) {
//					List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
//					double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
//					currentNetworkOutputs.add(currentNetworkOutput[0]);
//					if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
//						System.out.println("Network output is NaN.");
//						System.exit(0);
//					}
//				}
//				// b. Add the highest network output to allHighestNetworkOutputs; it does not matter whether it
//				// appears more than once (if this happens, it is solved within determineBestVoiceAssignment() -- which
//				// is called in training, test, and application case)
//				double currentHighestNetworkOutput = Collections.max(currentNetworkOutputs);
////				if (Collections.frequency(currentNetworkOutputs, currentHighestNetworkOutput) > 1) {
////				 	System.out.println("Highest network output appears more than once.");
////					System.exit(0);
////				}
//				allHighestNetworkOutputs.add(currentHighestNetworkOutput);
//
//				// c. Determine the best voice assignment and add it to allBestVoiceAssignments. Because it is possible 
//				// that different voice assignments result in the same network output, the highest output may occur
//				// multiple times. If this is the case, less likely candidates must be filtered out; this happens 
//				// inside determineBestVoiceAssignment()
//				List<Integer> predictedBestVoiceAssignment = 
////					outputEvaluator.determineBestVoiceAssignment(currentBasicTabSymbolPropertiesChord, currentNetworkOutputs,
////					currentPossibleVoiceAssignments, isTrainingOrTestMode); TODO <-- veranderd voor SysMus
//					outputEvaluator.determineBestVoiceAssignment(currentNetworkOutputs, currentPossibleVoiceAssignments); 
//				allBestVoiceAssignments.add(predictedBestVoiceAssignment);
//
//				// d. Convert predictedBestVoiceAssignment into a List of voices, and add it to allPredictedVoices
//				List<List<Double>> predictedChordVoiceLabels = dataConverter.getChordVoiceLabels(predictedBestVoiceAssignment); 
//				List<List<Integer>> predictedChordVoices = dataConverter.getVoicesInChord(predictedChordVoiceLabels); 
//				allPredictedVoices.addAll(predictedChordVoices);
//			}
//	    }
//	    allPredictedVoicesAndDurations.add(allPredictedVoices);
//	    allPredictedVoicesAndDurations.add(allPredictedDurations);
//	    return allPredictedVoicesAndDurations; 
//	}


	public void divideTrainingAndTestSets(int num) {
	}


	public void createCVSets(List<List<List<Double>>> inputs, List<List<List<Double>>> outputs) {
	}


	/**
	 * Creates groundTruthVoiceLabels.
	 * 
	 * @param argGroundTruthChordVoiceLabels
	 */
	private List<List<Double>> createGroundTruthVoiceLabels(List<List<List<Double>>> argGroundTruthChordVoiceLabels) {
		List<List<Double>> gtVoiceLabels;
//		groundTruthVoiceLabels = new ArrayList<List<Double>>();
		gtVoiceLabels = new ArrayList<List<Double>>();
		for (List<List<Double>> l : argGroundTruthChordVoiceLabels) {
//			groundTruthVoiceLabels.addAll(l);	
			gtVoiceLabels.addAll(l);
		}
		return gtVoiceLabels;
	}


	/**
	 * Creates groundTruthVoiceLabels.
	 * 
	 * @param argGroundTruthVoiceLabels Is <code>null</code> in C2C case
	 * @param argGroundTruthChordVoiceLabels Is <code>null</code> in N2N case
	 */
	private List<List<Double>> createGroundTruthVoiceLabelsOUD(List<List<Double>> argGroundTruthVoiceLabels, 
		List<List<List<Double>>> argGroundTruthChordVoiceLabels) {
		
		List<List<Double>> gtVoiceLabels;
		
		// Verify that one of the arguments is null and the other not
		if ((argGroundTruthVoiceLabels != null && argGroundTruthChordVoiceLabels != null) || 
			(argGroundTruthVoiceLabels == null && argGroundTruthChordVoiceLabels == null)) {
			System.out.println("ERROR: if argGroundTruthVoiceLabels == null, "
				+ "argGroundTruthChordVoiceLabels must not be, and vice versa" + "\n");
			throw new RuntimeException("ERROR (see console for details)");
		}
		
		// N2N
		if (argGroundTruthVoiceLabels != null) {
//			groundTruthVoiceLabels = argGroundTruthVoiceLabels;
			gtVoiceLabels = argGroundTruthVoiceLabels;
		}
		// C2C
		else {
//			groundTruthVoiceLabels = new ArrayList<List<Double>>();
			gtVoiceLabels = new ArrayList<List<Double>>();
			for (List<List<Double>> l : argGroundTruthChordVoiceLabels) {
//				groundTruthVoiceLabels.addAll(l);	
				gtVoiceLabels.addAll(l);
			}
		}
		return gtVoiceLabels;
	}

	
	// BELOW THIS LINE: obsolete methods
	/**
	 * For a single training run, stores the training settings, the voice labels, and the network's output in
	 * a .txt file.
	 * 
	 * @param modelParameters
	 * @param run
	 * @param assignmentErrors
	 * @param bestPredictedVoices
	 * @param path
	 * @param bestWeights
	 */
	// FIXME commented out 12.11.2016
	private String getTrainingResultsOLD(Map<String, Double> modelParameters,
//	public void storeTrainingRecordCurrentRun(Map<String, Double> modelParameters,
		String trainingStartTime,
		/*double smallestNetworkError, List<List<Integer>> assignmentErrors, 
		List<List<Integer>> bestPredictedVoices, String path, 
		List<Double> bestWeights, boolean includeAdditionalMetrics,*/ 
		String allTrainingErrors, String errorSpecifications, 
		String trainingSettings/*, List<Integer[]> argVoicesCoDNotes*/) {	
//
////		String trainingStartTime = info[0];
////		String path = info[1];
////		String trainingSettings = info[2];
//		
//		// Get the training settings
////		List<String> trainingSetPieceNames = trainingPieceNames;
////		if (storeAdditionalFiles) {
////			trainingSetPieceNames = AuxiliaryTool.getStoredObject(new ArrayList<String>(), 
////				new File(path + trainingSetPieceNamesSuffix));
////		}
//
////		String settings = getTrainingSettings(run, argTrainingInfoMap, storedTrainingSetPieceNames); 
////		String settings =
////			"==========================================" + "\r\n" +
////			"Training started on " + trainingStartTime + "\r\n" +   
////			"Training completed on " + trainingEndTime + "\r\n" +
////			"==========================================" + "\r\n" + "\r\n";
//
////		settings = settings.concat(getTrainingSettings(/*run,*/ modelParameters, 
////			trainingSetPieceNames, path, true));
//		String settings = "";
//		settings = settings.concat(trainingSettings);
//
//		// Get the label and network output details
////		int highestNumberOfVoicesTraining = modelParameters.get(HIGHEST_NUMBER_OF_VOICES).intValue();
////		boolean argModelDuration = false;
////		if (!isBidirectional) {
////			if (modelDuration) {
////				argModelDuration = true;
////			}
////		}
////		if (isBidirectional) {
////			if (modelDuration && modelDurationAgain) {
////				argModelDuration = true;
////			}
////		}
////		String errorSpecifications = errorCalculator.getErrorSpecifications(assignmentErrors, 
////			smallestNetworkError, bestPredictedVoices, groundTruthVoiceLabels, equalDurationUnisonsInfo, 
////			highestNumberOfVoicesTraining, true, argModelDuration);
//
////		// Insert network error into errorSpecifications
////		int splitIndex = errorSpecifications.indexOf("  number of notes assigned to the incorrect voice");
////		String firstHalf = errorSpecifications.substring(0, splitIndex);
////		String secondHalf = errorSpecifications.substring(splitIndex, errorSpecifications.length());
////		errorSpecifications = firstHalf.concat("accompanying network error: " + smallestNetworkError + "\r\n");
////		errorSpecifications = errorSpecifications.concat(secondHalf);
//////	firstHalf + "accompanying network error: " + smallestNetworkError + "\r\n" + secondHalf;
//
//		List<List<Integer>> conflictIndices = null;
//		List<List<Integer>> allPredictedVoices = null;
//		List<List<Double>> allPredictedDurationLabels = null;
//		List<Rational[]> allMetricPositions = null;
//		List<Integer> backwardsMapping = null;
////		String details = getActualAndPredictedVoicesDetails(argTrainingInfoMap, bestWeights, conflictIndices, 
////			bestPredictedVoices, predictedDurationLabels, allMetricPositions, backwardsMapping); // MEDREN
//		String details = getActualAndPredictedVoicesDetails(modelParameters, /*bestWeights,*/ 
//			conflictIndices, allPredictedVoices, allPredictedDurationLabels, allMetricPositions, 
//			/*argVoicesCoDNotes,*/ backwardsMapping);//, chordSizes, allNetworkOutputs, groundTruthVoiceLabels, 
////			equalDurationUnisonsInfo, groundTruthDurationLabels, voicesCoDNotes, 
////			groundTruthChordVoiceLabels, possibleVoiceAssignmentsAllChords, 
////			allBestVoiceAssignments, allHighestNetworkOutputs);
//
//		// Combine into trainingRecord
//		// Make the header, containing the training end time, and combine into trainingRecord
//		String header = 
//			"==========================================" + "\r\n" +
//			"training started   : " + trainingStartTime + "\r\n" +   
//			"training completed : " + ToolBox.getTimeStamp() + "\r\n" +
//			"==========================================" + "\r\n" + "\r\n";
//		
//		return header + settings + errorSpecifications + details + allTrainingErrors;
//
////		// Add training errors
////		trainingRecord = trainingRecord.concat(allTrainingErrors);
//
//		// Make the fileName and store trainingRecord
////		int learningApproach = argTrainingInfoMap.get(LEARNING_APPROACH).intValue();
////		String learningApproachAsString = null;
////		if (learningApproach == NOTE_TO_NOTE) {
////			learningApproachAsString = "(N)";
////		}
////		else if (learningApproach == CHORD_TO_CHORD) {
////			learningApproachAsString = "(C)";
////		}
////		String fileName = path + "Training process record "+ learningApproachAsString + ".txt";
////		String fileName = path + trainRec + ".txt";
////		if (storeAdditionalFiles) {
////			fileName = path + "Training process record "+ learningApproachAsString + " run " + run + ".txt";
////		}
//
////		AuxiliaryTool.storeTextFile(trainingRecord, new File(fileName));
////		AuxiliaryTool.storeTextFile(trainingRecord, new File(path + trainRec + ".txt"));
		return null;
	}
	
	
//	/**
//	 * For a single training run, stores the training settings, the voice labels, and the network's output in
//	 * a .txt file.
//	 * 
//	 * @param modelParameters
//	 * @param run
//	 * @param assignmentErrors
//	 * @param bestPredictedVoices
//	 * @param path
//	 * @param bestWeights
//	 */
//	public void storeTrainingRecordCurrentRunZOALSHETWAS(Map<String, Double> modelParameters, /*int run,*/ 
//		double smallestNetworkError, List<List<Integer>> assignmentErrors, 
//		List<List<Integer>> bestPredictedVoices, String path, List<Double> bestWeights/*, 
//		boolean includeAdditionalMetrics*/, String allTrainingErrors) {	
//		
//		// Get the training settings
//		List<String> trainingSetPieceNames = trainingPieceNames;
////		if (storeAdditionalFiles) {
////			trainingSetPieceNames = AuxiliaryTool.getStoredObject(new ArrayList<String>(), 
////				new File(path + trainingSetPieceNamesSuffix));
////		}
//		
////		  String settings = getTrainingSettings(run, argTrainingInfoMap, storedTrainingSetPieceNames); 
//		String settings =
//			"==========================================" + "\r\n" +
//			"Training started on " + trainingStartTime + "\r\n" +   
//			"Training completed on " + trainingEndTime + "\r\n" +
//			"==========================================" + "\r\n" + "\r\n";
//		
//		settings = settings.concat(getTrainingSettings(/*run,*/ modelParameters, 
//			trainingSetPieceNames, path, true));
//
//	  // Get the label and network output details
//	  int highestNumberOfVoicesTraining = modelParameters.get(HIGHEST_NUMBER_OF_VOICES).intValue();
//	  boolean argModelDuration = false;
//	  if (!isBidirectional) {
//			if (modelDuration) {
//				argModelDuration = true;
//			}
//		}
//		if (isBidirectional) {
//			if (modelDuration && modelDurationAgain) {
//				argModelDuration = true;
//			}
//		}
//	  String errorSpecifications = errorCalculator.getErrorSpecifications(assignmentErrors, 
//			bestPredictedVoices, groundTruthVoiceLabels, equalDurationUnisonsInfo, 
//			highestNumberOfVoicesTraining, true, argModelDuration);
//	  
//	  // Insert network error into errorSpecifications
//	  int splitIndex = errorSpecifications.indexOf("  number of notes assigned to the incorrect voice");
//	  String firstHalf = errorSpecifications.substring(0, splitIndex);
//	  String secondHalf = errorSpecifications.substring(splitIndex, errorSpecifications.length());
//	  errorSpecifications = firstHalf.concat("accompanying network error: " + smallestNetworkError + "\r\n");
//	  errorSpecifications = errorSpecifications.concat(secondHalf);
////			  firstHalf + "accompanying network error: " + smallestNetworkError + "\r\n" + secondHalf;
//	  
//	  List<List<Integer>> conflictIndices = null;
//	  List<List<Integer>> predictedVoices = null;
//	  List<List<Double>> predictedDurationLabels = null;
//	  List<Rational[]> allMetricPositions = null;
//	  List<Integer> backwardsMapping = null;
////	  String details = getActualAndPredictedVoicesDetails(argTrainingInfoMap, bestWeights, conflictIndices, 
////	  	bestPredictedVoices, predictedDurationLabels, allMetricPositions, backwardsMapping); // MEDREN
//	  String details = getActualAndPredictedVoicesDetails(modelParameters, bestWeights, conflictIndices, 
//		  predictedVoices, predictedDurationLabels, allMetricPositions, backwardsMapping); // MEDREN
//	  
//	  // Combine into trainingRecord
//	  String trainingRecord = settings + errorSpecifications + details;
//	  
//	  // Add training errors
//	  trainingRecord = trainingRecord.concat(allTrainingErrors);
//			  
//		// Make the fileName and store trainingRecord
////	  int learningApproach = argTrainingInfoMap.get(LEARNING_APPROACH).intValue();
////	  String learningApproachAsString = null;
////		if (learningApproach == NOTE_TO_NOTE) {
////			learningApproachAsString = "(N)";
////		}
////		else if (learningApproach == CHORD_TO_CHORD) {
////			learningApproachAsString = "(C)";
////		}
////		String fileName = path + "Training process record "+ learningApproachAsString + ".txt";
//		String fileName = path + trainRec + ".txt";
////		if (storeAdditionalFiles) {
////			fileName = path + "Training process record "+ learningApproachAsString + " run " + run + ".txt";
////		}
//		
//		AuxiliaryTool.storeTextFile(trainingRecord, new File(fileName));
//	}
	
	
//	/**
//	 * Stores the training settings and information on the classification errors and their accompanying (smallest)
//	 * training errors in a .txt file when the training (for the given number of training runs) is completed.
//	 * 
//	 * @param argTrainingInfoMap
//	 * @param smallestClassificationErrors
//	 * @param allSmallestAssignmentErrors
//	 * @param allBestPredictedVoices
//	 * @param smallestTrainingErrors
//	 * @param path
//	 */
//	private void storeTrainingResults(Map<String, Double> argTrainingInfoMap,
//		List<Double> smallestClassificationErrors, List<List<List<Integer>>> allSmallestAssignmentErrors, 
//		List<List<List<Integer>>> allBestPredictedVoices, List<Double> smallestTrainingErrors, String path) {
//		/*, boolean includeAdditionalMetrics) {*/		
//	  			
//		String settings = 
//			"=======================================" + "\r\n" +
//		  "Training started   " + trainingStartTime + "\r\n" +   
//		  "Training completed " + trainingEndTime + "\r\n" +
//		  "=======================================" + "\r\n" + "\r\n";
//			
//		int learningApproach = argTrainingInfoMap.get(MODELLING_APPROACH).intValue();
////		int numberOfRuns = argTrainingInfoMap.get(NUMBER_OF_RUNS).intValue();
//		int numberOfRuns = -1;
//		int highestNumberOfVoicesTraining = argTrainingInfoMap.get(HIGHEST_NUMBER_OF_VOICES).intValue();
//		
//		// Get the training settings; set run to -1 as it does not apply here
//		int run = -1;
//		List<String> trainingSetPieceNames = trainingPieceNames;
////		if (storeAdditionalFiles) {
////			trainingSetPieceNames = AuxiliaryTool.getStoredObject(new ArrayList<String>(), 
////				new File(path + trainingSetPieceNamesSuffix));
////		}
//	  settings = settings.concat(getTrainingSettings(/*run,*/ argTrainingInfoMap, 
//			trainingSetPieceNames, path, true));
//	  String learningApproachAsString = null;
//		if (learningApproach == NOTE_CLASS) {
//			learningApproachAsString = "N";
//		}
//		else if (learningApproach == CHORD_REGRESS) {
//			learningApproachAsString = "C";
//		}
//		
//		settings = settings.concat("ERRORS" + "\r\n");
//		settings = settings.concat("smallest classification errors with accompanying (smallest) network error for " + numberOfRuns
//			+ " runs: " + "\r\n");
//
//		// For each run, list the smallest classification error		
//		for (int i = 0; i < smallestClassificationErrors.size(); i++) {
//			double currentSmallestClassificationError = smallestClassificationErrors.get(i);
//			settings = settings.concat("  run " + i + ": " + currentSmallestClassificationError + " (network error = " +
//		    smallestTrainingErrors.get(i) + ")" + "\r\n");
//		}
//		System.out.println("smallestClassificationErrors = " + smallestClassificationErrors);
//	  
//		// Get the smallest classification error of all runs 
//		double smallestClassificationErrorOfAllRuns = Collections.min(smallestClassificationErrors);
//		System.out.println("smallestClassificationErrorOfAllRuns = " + smallestClassificationErrorOfAllRuns);
//		
//		// Check whether smallestClassificationErrorOfAllRuns occurs multiple times in the list: if so, find the run 
//		// with the one with the smallest accompanying training error.
//		// 1. List the runs with smallestClassificationErrorOfAllRuns 
//		List<Integer> runsWithSmallestClassificationError = new ArrayList<Integer>();
//		for (int i = 0; i < smallestClassificationErrors.size(); i++) {
//		  if (smallestClassificationErrors.get(i) == smallestClassificationErrorOfAllRuns) {
//		    runsWithSmallestClassificationError.add(i);
//		  }
//		}
//		settings = settings.concat("runs with smallest classification error: " + 
//		  runsWithSmallestClassificationError + "\r\n"); 		
//		
//		// 2. If smallestClassificationErrorOfAllRuns occurs multiple times: find the smallest accompanying training error 
//		// and reset runWithSmallestClassificationError. If smallestClassificationError occurs only once,
//		// runWithSmallestClassificationError will be the first and only element of runsWithSmallestClassificationError.
//		int runWithSmallestClassificationError = runsWithSmallestClassificationError.get(0);
//		if (runsWithSmallestClassificationError.size() > 1) {
//		  // a. Set smallestAccompanyingTrainingError to the value of the training error that accompanies the first
//		  // element of runsWithSmallestClassificationError
//		  double smallestAccompanyingTrainingError = smallestTrainingErrors.get(runWithSmallestClassificationError); 
//		  // b. Compare with the values of the next accompanying training error(s); if a smaller value is found,
//		  // reset smallestAccompanyingTrainingError and runWithSmallestClassificationError
//		  for (int i = 1; i < runsWithSmallestClassificationError.size(); i++) {
//		    int currentRunWithSmallestClassificationError = runsWithSmallestClassificationError.get(i);
//		    double currentAccompanyingTrainingError = smallestTrainingErrors.get(currentRunWithSmallestClassificationError); 
//		    if (currentAccompanyingTrainingError < smallestAccompanyingTrainingError) {
//		      smallestAccompanyingTrainingError = currentAccompanyingTrainingError;
//		      runWithSmallestClassificationError = currentRunWithSmallestClassificationError;
//		    }
//		  }
//		}
//		settings = settings.concat("smallest classification error (with the smallest accompanying network error) obtained in " +
//		  "run no. " + runWithSmallestClassificationError + ":" + "\r\n" +
//		  "  classification error: " + smallestClassificationErrorOfAllRuns + "\r\n" +
//		  "  accompanying network error: " + smallestTrainingErrors.get(runWithSmallestClassificationError) + "\r\n \r\n");
//		
//    // Get the corresponding elements from allSmallestAssignmentErrors, get the error specifications and add them
//		// to settings
// 		List<List<Integer>> smallestAssignmentErrors = allSmallestAssignmentErrors.get(runWithSmallestClassificationError);		
//    List<List<Integer>> bestPredictedVoices = allBestPredictedVoices.get(runWithSmallestClassificationError);
// 		//		String errorSpecifications = getErrorSpecifications(smallestAssignmentErrors, true, argErrorCalculator);
//    boolean argModelDuration = false;
//	  if (!isBidirectional) {
//			if (modelDuration) {
//				argModelDuration = true;
//			}
//		}
//		if (isBidirectional) {
//			if (modelDuration && modelDurationAgain) {
//				argModelDuration = true;
//			}
//		}	
//    String errorSpecifications = errorCalculator.getErrorSpecifications(smallestAssignmentErrors,
// 			bestPredictedVoices, groundTruthVoiceLabels, equalDurationUnisonsInfo, highestNumberOfVoicesTraining, 
// 			true, argModelDuration);
// 		settings = settings.concat(errorSpecifications);
//		
//	  // Rename the best weights from the run with the smallest classification error for the test and application case
//		// 1. Construct the appropriate filename and get the File
////		String bestWeightsRunWithSmallestClassErrFileName = null;
//		String bestWeightsRunWithSmallestClassErrFileName = bestWeightsFileName + " run " +
//			    runWithSmallestClassificationError + ".xml"; 
////		if (learningApproach == NOTE_TO_NOTE) {
////			bestWeightsRunWithSmallestClassErrFileName = bestWeightsFileNoteToNotePrefix + " run " +
////		    runWithSmallestClassificationError + ".xml";
////		}
////		else if (learningApproach == CHORD_TO_CHORD) {
////			bestWeightsRunWithSmallestClassErrFileName = bestWeightsFileChordToChordPrefix + " run " + 
////			  runWithSmallestClassificationError + ".xml";
////		}
//		File bestWeightsRunWithSmallestClassErrFile = new File(path + bestWeightsRunWithSmallestClassErrFileName); 
//	  // 2. Shorten the filename by removing the run information from it
//		String newFileName = 
//	   	bestWeightsRunWithSmallestClassErrFileName.replace(" run " + runWithSmallestClassificationError + ".xml", ".xml");
////		System.out.println(newFileName);
//	  // 3. Use the new file name to make the new File
//	  File newFile = new File(path + newFileName); 
////	  System.out.println("newFile name =     " + newFile.getAbsolutePath());
//	  // 4. If there already exists a File with the new name: remove it
//	  File currentFolder = new File(path);
//	  String[] filesInCurrentPath = currentFolder.list();
//	  for (int i = 0; i < filesInCurrentPath.length; i++) {
////	  	System.out.println(filesInCurrentPath[i]);
//	  	if (filesInCurrentPath[i].equals(newFileName)) {
//	  		newFile.delete();
//	    }
//	  }
//	  // 5. Rename the old file
//	  bestWeightsRunWithSmallestClassErrFile.renameTo(newFile);
//	    
//	  // Print out and store settings
//		String fileName = path + "Training results " + learningApproachAsString + ".txt";
//		System.out.println(settings);
//		AuxiliaryTool.storeTextFile(settings, new File(fileName));
//	}


//	/**
//	 * 
//	 * @param run
//	 * @param modelParameters
//	 * @param argTrainingSetPieceNames
//	 * @return
//	 */
//	private String getTrainingSettings(/*int run,*/ Map<String, Double> modelParameters,	
//		List<String> argTrainingSetPieceNames, String path, boolean isTraining) {
//		String trainingSettings = ""; 
//		
////		for (Map.Entry<String, Double> entry : modelParameters.entrySet()) {
////			System.out.println(entry.getKey() + " = " + entry.getValue());
////		}
////		System.exit(0);
//		
////		int numberOfRuns = argTrainingInfoMap.get(NUMBER_OF_RUNS).intValue();
//		int learningApproach = modelParameters.get(LEARNING_APPROACH).intValue(); 
//	  
//		String model = null; 
//		if (learningApproach == NOTE_TO_NOTE) {
//			if (!isBidirectional) {
//				model = "N";		
//			}
//			else {
//				model = "B";
//			}
//			if (modelDuration) {
//				model = model.concat("_prime");
//			}
//		}
//		else if (learningApproach == CHORD_TO_CHORD) {
//			model = "C";
//		}
//		
//		trainingSettings = "MODEL PARAMETERS AND DATA" + "\r\n";
//		trainingSettings = trainingSettings.concat("model = " + model + "\r\n");
//		trainingSettings = trainingSettings.concat("model backward = " + 
//			AuxiliaryTool.toBoolean(modelParameters.get(MODEL_BACKWARD).intValue()) + "\r\n");
//		trainingSettings = trainingSettings.concat("model duration = " + 
//			AuxiliaryTool.toBoolean(modelParameters.get(MODEL_DURATION).intValue()) + "\r\n");
//		trainingSettings = trainingSettings.concat("bidirectional = " + 
//			AuxiliaryTool.toBoolean(modelParameters.get(IS_BIDIRECTIONAL).intValue()) + "\r\n");
//		trainingSettings = trainingSettings.concat("\r\n");
//		//
//		trainingSettings = trainingSettings.concat("allow single-note unisons = " + 
//			AuxiliaryTool.toBoolean(modelParameters.get(ALLOW_COD).intValue()));
//		trainingSettings = trainingSettings.concat("deviation threshold = " + 
//			modelParameters.get(DEVIATION_THRESHOLD) + "\r\n");
//		trainingSettings = trainingSettings.concat("metacycles = " + 
//			modelParameters.get(MAX_META_CYCLES).intValue() + "\r\n");
//		trainingSettings = trainingSettings.concat("cycles = " + 
//			modelParameters.get(CYCLES).intValue() + "\r\n"); 	
//		trainingSettings = trainingSettings.concat("alpha = " + 
//			modelParameters.get(LEARNING_RATE) + "\r\n");
//		trainingSettings = trainingSettings.concat("lambda = " + 
//			modelParameters.get(REGULARISATION_PARAMETER) + "\r\n");
//		int hnf = modelParameters.get(HIDDEN_NEURONS_FACTOR).intValue();
//		String hnfAsString;
//		if (hnf < 0) {
//			hnfAsString = "1/" + (-1*hnf);
//		}
//		else {
//			hnfAsString = "" + hnf;
//		}
//		trainingSettings = trainingSettings.concat("hidden layer size = " + 
//			hnfAsString + " * input layer size" + "\r\n");
//		if (learningApproach == CHORD_TO_CHORD) {
//			trainingSettings = trainingSettings.concat("epsilon = " + 
//				modelParameters.get(MARGIN) + "\r\n");
//		}
//		//
//		trainingSettings = trainingSettings.concat("\r\n");
//		trainingSettings = trainingSettings.concat("training set =" + "\r\n");
//		for (int i = 0; i < argTrainingSetPieceNames.size(); i++) {
//			trainingSettings = trainingSettings.concat("    " + 
//				argTrainingSetPieceNames.get(i) + "\r\n");
//		}
//		int numTrainEx = -1;
//		String nOrC = "(notes)";
//		if (learningApproach == CHORD_TO_CHORD) {
//			nOrC = "(chords)";
//		}
//		if (isTraining) {
//			if (learningApproach == NOTE_TO_NOTE) {
//				numTrainEx = noteFeatures.size();
//			}
//			else if (learningApproach == CHORD_TO_CHORD) {
//				numTrainEx = chordFeatures.size();
//			}
//		}
//		// Test: get numTrainEx from already created training_rec file
//		else {
//			String contents = AuxiliaryTool.readTextFile(new File(path + trainRec + ".txt"));
//			String s = AuxiliaryTool.getAllowedCharactersAfterMarker(contents,
//				("number of training examples " + nOrC + " = "));
//			numTrainEx = Integer.parseInt(s);
//		}
//		trainingSettings = trainingSettings.concat("number of training examples " + 
//			nOrC + " = " + numTrainEx + "\r\n");
//		trainingSettings = trainingSettings.concat("largest chord in training set = " + 
//			modelParameters.get(LARGEST_CHORD_SIZE).intValue() + "\r\n");
//		trainingSettings = trainingSettings.concat("highest number of voices in training set = " + 
//			modelParameters.get(HIGHEST_NUMBER_OF_VOICES).intValue() + "\r\n");
//		trainingSettings = trainingSettings.concat("\r\n");
//		
//		return trainingSettings;
//		
////		String featureSetAsString = null;
////		if (argTrainingInfoMap.get(FEATURE_SET) == FeatureGenerator.FEATURE_SET_A) { 
////			featureSetAsString = "FEATURE_SET_A";
////		}
////		else if (argTrainingInfoMap.get(FEATURE_SET) == FeatureGenerator.FEATURE_SET_B) {
////			featureSetAsString = "FEATURE_SET_B";
////		}
////		else if (argTrainingInfoMap.get(FEATURE_SET) == FeatureGenerator.FEATURE_SET_C) { 
////			featureSetAsString = "FEATURE_SET_C";
////		}
////		else if (argTrainingInfoMap.get(FEATURE_SET) == FeatureGenerator.FEATURE_SET_D) {
////			featureSetAsString = "FEATURE_SET_D";
////		}
//				
////		String allowCoDAsString = null;
////		if (modelParameters.get(ALLOW_COD) == COD_NOT_ALLOWED) {
////			allowCoDAsString = "false";
////		}
////		else if (modelParameters.get(ALLOW_COD) == COD_ALLOWED) {
////			allowCoDAsString = "true";
////		}
//		
////		String errorMeasurementAsString = null;
////		if (argTrainingInfoMap.get(ERROR_MEASUREMENT).intValue() == TOLERANT) {
////			errorMeasurementAsString = "TOLERANT";
////		}
////		else if (argTrainingInfoMap.get(ERROR_MEASUREMENT).intValue() == STRICT) {
////			errorMeasurementAsString = "STRICT";
////		}
//		
////		String weightsInitialisationMethodAsString = null;
////		if (argTrainingInfoMap.get(WEIGHTS_INIT_METHOD) == RANDOM_INIT) {
////			weightsInitialisationMethodAsString = "RANDOM_INIT";
////		}
////		else if (argTrainingInfoMap.get(WEIGHTS_INIT_METHOD) == INIT_FROM_LIST) {
////			weightsInitialisationMethodAsString = "INIT_FROM_LIST";
////		}
////		else if (argTrainingInfoMap.get(WEIGHTS_INIT_METHOD) == INIT_FROM_FILE) {
////			weightsInitialisationMethodAsString = "INIT_FROM_FILE";
////		}
//				
////		boolean modelDuration = false;
////		if (modelParameters.get(MODEL_DURATION).intValue() == 1) {
////			modelDuration = true;
////		}
////		boolean modelBackward = false;
////		if (modelParameters.get(MODEL_BACKWARD).intValue() == 1) {
////			modelBackward = true;
////		}
////		boolean modelMelody = false;
////		if (modelParameters.get(MODEL_MELODY).intValue() == 1) {
////			modelMelody = true;
////		}
////		boolean isBidirectional = false;
////		if (modelParameters.get(IS_BIDIRECTIONAL).intValue() == 1) {
////			isBidirectional = true;
////		}
////		boolean modelDurationAgain = false;
////		if (modelParameters.get(MODEL_DURATION_AGAIN).intValue() == 1) {
////			modelDurationAgain = true;
////		}
////		trainingSettings = trainingSettings.concat("model backward = " + modelBackward + "\r\n");
////		trainingSettings = trainingSettings.concat("model duration = " + modelDuration + "\r\n");
////		trainingSettings = trainingSettings.concat("modelMelody = " + modelMelody + "\r\n");
////		trainingSettings = trainingSettings.concat("isBidirectional = " + isBidirectional + "\r\n");
////		trainingSettings = trainingSettings.concat("  modelDurationAgain = " + modelDurationAgain + "\r\n");
//		
////		if (path != null) {
////			File f = new File(path + "Training process record " + learningApproachAsString + ".txt");
////			String contents = AuxiliaryTool.readTextFile(f)
////		}
//
////		String numTrainingExamples = "";
////		number of training examples (notes) =
////		System.out.println(path);
//		
////		Integer in = null;
////		if (learningApproach == NOTE_TO_NOTE) {
////			trainingSettings = trainingSettings.concat("number of training examples (notes) = " + 
//////		    argTrainingInfoMap.get(NUMBER_OF_TRAINING_EXAMPLES).intValue() + "\r\n");
////			noteFeatures.size() + "\r\n");
////			trainingSettings = trainingSettings.concat("number of training examples (notes) = " +
////				AuxiliaryTool.getStoredObject(in, new File(path + numberOfTrainingExamplesNoteToNoteSuffix)) 
////				+ "\r\n");
////			trainingSettings = trainingSettings.concat("number of training examples (notes) = " 
////				+ numTrainEx + "\r\n");
////		}
////		else {
////			trainingSettings = trainingSettings.concat("number of training examples (chords) = " + 
//////		    argTrainingInfoMap.get(NUMBER_OF_TRAINING_EXAMPLES).intValue() + "\r\n");
////			chordFeatures.size() + "\r\n");
////			trainingSettings = trainingSettings.concat("number of training examples (chords) = " +
////				AuxiliaryTool.getStoredObject(in, new File(path + numberOfTrainingExamplesChordToChordSuffix)) 
////				+ "\r\n");
////			trainingSettings = trainingSettings.concat("number of training examples (chords) = " + 
////				numTrainEx + "\r\n");
////		}
//		  
////		if (learningApproach == NOTE_TO_NOTE) { //2016
////			if (!isNewModel) {
////			  trainingSettings = trainingSettings.concat("  featureSet = " + featureSetAsString + "\r\n");
////			}
////		}
//
////		if (learningApproach == NOTE_TO_NOTE) {
////			trainingSettings = trainingSettings.concat("  allow single-note unisons = " + allowCoDAsString + "\r\n");
////		  if (modelParameters.get(ALLOW_COD) == COD_ALLOWED) {
////		  	trainingSettings = trainingSettings.concat("  deviation threshold = " + 
////		      modelParameters.get(DEVIATION_THRESHOLD) + "\r\n");
////		  }
////		}
//		
////		trainingSettings = trainingSettings.concat("weights initialisation method = " + 
////			weightsInitialisationMethodAsString + "\r\n");
////		if (run == -1) {
////			trainingSettings = trainingSettings.concat("numberOfRuns = " + numberOfRuns + "\r\n");
////		}
////		if (numberOfRuns == -1) {
////			trainingSettings = trainingSettings.concat("run = " + run + "\r\n");
////		}	
////		trainingSettings = trainingSettings.concat("metacycles = " + 
////		  modelParameters.get(MAX_META_CYCLES).intValue() + "\r\n");
////		trainingSettings = trainingSettings.concat("cycles = " + modelParameters.get(CYCLES).intValue() + "\r\n"); 	
////		trainingSettings = trainingSettings.concat("alpha = " + modelParameters.get(LEARNING_RATE) + "\r\n");
////		trainingSettings = trainingSettings.concat("lambda = " + modelParameters.get(REGULARISATION_PARAMETER) + "\r\n");
////		if (learningApproach == CHORD_TO_CHORD) {
////			trainingSettings = trainingSettings.concat("epsilon = " + modelParameters.get(MARGIN) + "\r\n");
////		}
////		trainingSettings = trainingSettings.concat("hiddenNeuronsFactor = " + 
////		  argTrainingInfoMap.get(HIDDEN_NEURONS_FACTOR).intValue() + "\r\n");
////		int hnf = modelParameters.get(HIDDEN_NEURONS_FACTOR).intValue();
////		String hnfAsString;
////		if (hnf < 0) {
////			hnfAsString = "1/" + (-1*hnf);
////		}
////		else {
////			hnfAsString = "" + hnf;
////		}
//////		trainingSettings = trainingSettings.concat("hiddenNeuronsFactor = " + 
//////			argTrainingInfoMap.get(HIDDEN_NEURONS_FACTOR).intValue() + "\r\n");
////		trainingSettings = trainingSettings.concat("hidden layer size = " + 
////			hnfAsString + " * input layer size\r\n");
//	}


	// FIXME commented out 12.11.2016
	private List<List<Integer>> getConflictIndices(List<List<Integer>> conflictIndicesLists, 
		List<List<Integer>> predictedVoices, List<Integer[]> argEqualDurationUnisonsInfo, 
		List<double[]> argAllNetworkOutputs, List<Integer> voicesPredictedInitially, 
		List<Integer> predictedVoicesAdapted, List<Integer> actualVoices, int i) {
//		
//		List<Integer> indicesOfCorrToCorr = conflictIndicesLists.get(0);
//		List<Integer> indicesOfCorrToIncorr = conflictIndicesLists.get(1);
//		List<Integer> indicesOfIncorrToCorr = conflictIndicesLists.get(2);
//		List<Integer> indicesOfIncorrToIncorr = conflictIndicesLists.get(3); 
//		
//		// Keep track of the conflict reassignments
//		boolean voicesPredictedInitiallyCorrectly = false;
//		boolean predictedVoicesAdaptedCorrectly = false;
//		boolean voicesPredictedInitiallyCorrectlyUpper = false;
//		boolean predictedVoicesAdaptedCorrectlyUpper = false;
//		
//		// a. In the tablature case, where equal duration unisons do not apply
////		if (equalDurationUnisonsInfo == null) {
//		if (argEqualDurationUnisonsInfo == null) {
//			voicesPredictedInitiallyCorrectly = 
//				errorCalculator.assertCorrectness(voicesPredictedInitially, actualVoices);
//			predictedVoicesAdaptedCorrectly = 
//				errorCalculator.assertCorrectness(predictedVoicesAdapted, actualVoices);
//		}
//		// b. In the non-tablature case, when the note at index i is not part of an EDU
////		else if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) == null) { // HIER OK
//		else if (argEqualDurationUnisonsInfo != null && argEqualDurationUnisonsInfo.get(i) == null) { // HIER OK
//			voicesPredictedInitiallyCorrectly = 
//				errorCalculator.assertCorrectness(voicesPredictedInitially, actualVoices);
//			predictedVoicesAdaptedCorrectly = 
//				errorCalculator.assertCorrectness(predictedVoicesAdapted, actualVoices);
//		}
//		// c. In the non-tablature case, when the note at index i is part of an EDU
////		else if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) != null) { // HIER OK
//		else if (argEqualDurationUnisonsInfo != null && argEqualDurationUnisonsInfo.get(i) != null) { // HIER OK
//			// Only if the note at index i is not the last note
////			if (i != noteFeatures.size() - 1) {
//			int numNotes = predictedVoices.size();
//			if (i != numNotes - 1) {
//				// Determine the predicted and adapted voices for the lower and upper EDUnotes (.get(0) can be used
//				// because in the non-tablature case there are no CoDs and the lists will contain only one element)
//				int voicePredInitiallyLowerNote = voicesPredictedInitially.get(0);
////				double[] outputNextNote = allNetworkOutputs.get(i + 1); // HIER OK
//				double[] outputNextNote = argAllNetworkOutputs.get(i + 1); // HIER OK
//				boolean allowCoD = false;
//				double deviationThreshold = -1;
//				int voicePredInitiallyUpperNote = outputEvaluator.interpretNetworkOutput(outputNextNote, allowCoD,
//				  	deviationThreshold).get(0).get(0);
//				int adaptedVoiceLowerNote = predictedVoicesAdapted.get(0);
//				int adaptedVoiceUpperNote = predictedVoices.get(i + 1).get(0); // HIER OK
//				// Determine for both EDUnotes whether the predicted and adapted voices are correct
//				List<Integer[]> predictedAndAdaptedVoices = new ArrayList<Integer[]>();
//				predictedAndAdaptedVoices.add(new Integer[]{voicePredInitiallyLowerNote, voicePredInitiallyUpperNote});
//				predictedAndAdaptedVoices.add(new Integer[]{adaptedVoiceLowerNote, adaptedVoiceUpperNote});
////				List<Integer> allowedVoices = Arrays.asList(new Integer[]{equalDurationUnisonsInfo.get(i)[1], 
////					equalDurationUnisonsInfo.get(i)[0]}); // HIER OK
//				List<Integer> allowedVoices = Arrays.asList(new Integer[]{argEqualDurationUnisonsInfo.get(i)[1], 
//					argEqualDurationUnisonsInfo.get(i)[0]}); // HIER OK
//				boolean[][] unisonNotesPredictedCorrectly = 
//					errorCalculator.assertCorrectnessEDUNotes(predictedAndAdaptedVoices, 
//					allowedVoices); 
//				boolean lowerUnisonNotePredictedCorrectly = unisonNotesPredictedCorrectly[0][0];
//				boolean upperUnisonNotePredictedCorrectly = unisonNotesPredictedCorrectly[0][1];
//				boolean lowerUnisonNoteAdaptedCorrectly = unisonNotesPredictedCorrectly[1][0];
//				boolean upperUnisonNoteAdaptedCorrectly = unisonNotesPredictedCorrectly[1][1];
//				// Set variables
//				voicesPredictedInitiallyCorrectly = lowerUnisonNotePredictedCorrectly;
//				predictedVoicesAdaptedCorrectly = lowerUnisonNoteAdaptedCorrectly;
//				voicesPredictedInitiallyCorrectlyUpper = upperUnisonNotePredictedCorrectly;
//				predictedVoicesAdaptedCorrectlyUpper = upperUnisonNoteAdaptedCorrectly;
//			}
//		}
//		// Determine correctness of reassignments
//		// 1. In the tablature case, and in the non-tablature case for non EDUnotes and lower EDUnotes
//		// a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
//		if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == true) {
//			indicesOfCorrToCorr.add(i); // HIER OK
//		}
//		// b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
//		if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == false) {
//			indicesOfCorrToIncorr.add(i); // HIER OK
//		}
//		// c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
//		if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == true) {
//			indicesOfIncorrToCorr.add(i); // HIER OK
//		}
//		// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
//		if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == false) {
//			indicesOfIncorrToIncorr.add(i); // HIER OK
//		}  	
//		// 2. In the non-tablature case for upper EDUnotes 
//		// NB: To avoid the index of the upper EDUnote being added twice (once when i is the index of the lower 
//		// EDUnote and once that of the upper), it must only be added when i is the index of the lower EDUnote
//		// TODO How is this implemented -- by means of the i + 1?
////		if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) != null) { // HIER OK
//		if (argEqualDurationUnisonsInfo != null && argEqualDurationUnisonsInfo.get(i) != null) { // HIER OK
//			// a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
//			if (voicesPredictedInitiallyCorrectlyUpper == true && predictedVoicesAdaptedCorrectlyUpper == true) {
//				if (!indicesOfCorrToCorr.contains(i + 1)) { // HIER OK
//					indicesOfCorrToCorr.add(i + 1); // HIER OK
//				}
//			}
//			// b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
//			if (voicesPredictedInitiallyCorrectlyUpper == true && predictedVoicesAdaptedCorrectlyUpper == false) {
//				if (!indicesOfCorrToIncorr.contains(i + 1)) { // HIER OK
//					indicesOfCorrToIncorr.add(i + 1); // HIER OK
//				}
//			}
//			// c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
//			if (voicesPredictedInitiallyCorrectlyUpper == false && predictedVoicesAdaptedCorrectlyUpper == true) {
//				if (!indicesOfIncorrToCorr.contains(i + 1)) { // HIER OK
//					indicesOfIncorrToCorr.add(i + 1); // HIER OK
//				}
//			}
//			// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
//			if (voicesPredictedInitiallyCorrectlyUpper == false && predictedVoicesAdaptedCorrectlyUpper == false) {
//				if (!indicesOfIncorrToIncorr.contains(i + 1)) { // HIER OK
//					indicesOfIncorrToIncorr.add(i + 1); // HIER OK
//				}
//			}
//		}
//		return conflictIndicesLists;
		return null;
	}


	/**
	 * Sets allHighestNetworkOutputs and allBestVoiceAssignments.
	 */
	// FIXME commented out 12.11.2016
	private void setAllHighestNetworkOutputsZOALSHETWAS() { 
//		int numberOfChords = chordFeatures.size();
//		// allHighestNetworkOutputs, as well as allBestVoiceAssignments, must be recreated every time 
//		// this method is called in the training case. In the test case, this method is only called once, so
//		// recreation is not strictly necessary -- but it is harmless nevertheless
//		allHighestNetworkOutputs = new ArrayList<Double>(); 
//		allBestVoiceAssignments = new ArrayList<List<Integer>>();
//		// For each chord
//		for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
//			// a. For all possible feature vectors for this chord: evaluate the network and
//			// add the result, the network output, to currentNetworkOutputs
//			List<List<Double>> currentChordFeatures = chordFeatures.get(chordIndex);
//			List<List<Integer>> currentPossibleVoiceAssignments = 
//				possibleVoiceAssignmentsAllChords.get(chordIndex);  	
//			
//			// TURNED INTO METHOD createNetworkOutputsChord() -->
//			List<Double> currentNetworkOutputs = new ArrayList<Double>();
//			for (int j = 0; j < currentChordFeatures.size(); j++) {
//				List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
//				double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
//				currentNetworkOutputs.add(currentNetworkOutput[0]);
//				if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
//					System.out.println("Network output is NaN.");
//					System.exit(0);
//				}
//			}
//			// TURN INTO METHOD <--
//			
//			// b. Add the highest network output to allHighestNetworkOutputs; it does not 
//			// matter whether it appears more than once (if this happens, it is solved 
//			// within determineBestVoiceAssignment(), which is called in training, test,
//			// and application case)
//			double currentHighestNetworkOutput = Collections.max(currentNetworkOutputs);
////			if (Collections.frequency(currentNetworkOutputs, currentHighestNetworkOutput) > 1) {
////			 	System.out.println("Highest network output appears more than once.");
////				System.exit(0);
////			}
//			allHighestNetworkOutputs.add(currentHighestNetworkOutput);
//
//			// c. Determine the best voice assignment and add it to allBestVoiceAssignments. Because it is possible 
//			// that different voice assignments result in the same network output, the highest output may occur
//			// multiple times. If this is the case, less likely candidates must be filtered out; this happens 
//			// inside determineBestVoiceAssignment()
//			List<Integer> predictedBestVoiceAssignment = 
////				outputEvaluator.determineBestVoiceAssignment(currentBasicTabSymbolPropertiesChord, currentNetworkOutputs,
////				currentPossibleVoiceAssignments, isTrainingOrTestMode);
//				outputEvaluator.determineBestVoiceAssignment(currentNetworkOutputs, currentPossibleVoiceAssignments); 
//			allBestVoiceAssignments.add(predictedBestVoiceAssignment);
//		}
	}


	/**
	 * Gets the results from the test or application process.
	 * 
	 * @param learningApproach
	 * @param assignmentErrors
	 * @param bestWeights
	 * @param info
	 * @param numberOfTestExamples
	 * @param allMetricPositions In fwd order when using bwd model
	 * @param conflictIndices In bwd order when using bwd model
	 * @param allPredictedVoices In bwd order when using bwd model
	 * @param allPredictedDurationLabels In fwd order when using bwd model
	 * @param backwardsMapping Is <code>null</code> when using the fwd model
	 * @return
	 */
	// FIXME commented out 12.11.2016
	private String getTestAndApplicationResultsNOTINUSE(Map<String, Double> modelParameters, 
		/*int learningApproach, List<List<Integer>> assignmentErrors, List<Double> bestWeights,*/ 
		String info[], int numberOfTestExamples, // TODO get otherwise in method 
		/*List<Rational[]> allMetricPositions,*/ List<List<Integer>> conflictIndices, 
		List<List<Integer>> allPredictedVoices, List<List<Double>> allPredictedDurationLabels, 
		/*List<Integer[]> argVoicesCoDNotes,*/ List<Integer> backwardsMapping, 
		String conflictsRecordTest, /*, List<List<Double>> argGroundTruthVoiceLabels*/
		String errorSpecifications) {

////		String results = "";	
//		String startTime = info[0];
////		String endTime = info[1];
//		String currentPath = info[1];
//		String pieceName = info[2];
//		String additionalInfo = info[3]; // is empty in test case, contains conflictsRecord in application case
//
//		// Get the trainingSetPieceNames and the trainingInfoMap, both of which were stored during the training			
////		List<String> trainingSetPieceNames = trainingPieceNames;
////		if (storeAdditionalFiles) {
////			trainingSetPieceNames = AuxiliaryTool.getStoredObject(new ArrayList<String>(), 
////				new File(currentPath + trainingSetPieceNamesSuffix));
////		}
//
////		Map<String, Double> storedTrainingInfoMap = null;
////		Map<String, Double> storedTrainingInfoMap = trainingParameters;
//		String numberOfTestExamplesAsString = null;
////		String paramPath = null;
////		// If cross-validation
////		if (currentPath.contains("Fold")) { // Find safer solution
////			paramPath = currentPath.substring(0, currentPath.indexOf("Fold"));
////		}
////		// If no cross-validation
////		else {
////			paramPath = currentPath.substring(0, currentPath.indexOf("No CV"));
////		}
//		int learningApproach = modelParameters.get(MODELLING_APPROACH).intValue();
//		if (learningApproach == NOTE_CLASS) {
////			storedTrainingInfoMap = AuxiliaryTool.getStoredObject(new HashMap<String, Double>(), new File(
//////			currentPath + trainingInfoMapNoteToNoteSuffix));
////			paramPath + trainingInfoMapNoteToNoteSuffix));
//			numberOfTestExamplesAsString = "number of test examples (notes) = " + numberOfTestExamples + "\r\n";
////			numberOfTestExamplesAsString = "number of test examples (notes) = " + getGroundTruthVoiceLabels().size() + "\r\n";
//		}
//		else if (learningApproach == CHORD_REGRESS) {
////			storedTrainingInfoMap = AuxiliaryTool.getStoredObject(new HashMap<String, Double>(), new File(
//////			currentPath + trainingInfoMapChordToChordSuffix));
////			paramPath + trainingInfoMapChordToChordSuffix));
//			numberOfTestExamplesAsString = "number of test examples (chords) = " + numberOfTestExamples + "\r\n";
////			numberOfTestExamplesAsString = "number of test examples (chords) = " + getGroundTruthChordVoiceLabels.size() + "\r\n";
//		}
//
////		// 1. Make header and add to results
////		results = results.concat("===================================" + "\r\n");
////		results = results.concat("Test started   " + startTime + "\r\n");   
////		results = results.concat("Test completed " + endTime + "\r\n");
////		results = results.concat("===================================" + "\r\n" + "\r\n");
//
//		String results;
//		
//		// 1. Make test settings and add to results
//		String testSettings = "TEST SETTINGS" + "\r\n";
//		testSettings = testSettings.concat("test set = " + "\r\n");
//		testSettings = testSettings.concat("  " + pieceName + "\r\n");
//		testSettings = testSettings.concat(numberOfTestExamplesAsString + "\r\n");
//		results = results.concat(testSettings);
//
//		// 2. Get training settings and add to results 
////		int run = -1;
////		String trainingSettings = getTrainingSettings(/*run,*/ storedTrainingInfoMap, 
////			trainingSetPieceNames, currentPath, false);
//		// 3. Get trainingSettings from .txt file created during training and add to results
//		String contents = 
//			ToolBox.readTextFile(new File(currentPath + trainRec + ".txt"));
////		String s = AuxiliaryTool.getAllowedCharactersAfterMarker(contents,
////			("number of training examples " + nOrC + " = "));
////		numTrainEx = Integer.parseInt(s);
//		String storedTrSettings = contents.substring(contents.indexOf(headerRec),
//			contents.indexOf(specRec));
//		results = results.concat(storedTrSettings);
//
//		// 3. Get error specifications and add to results
////		int highestNumberOfVoicesTraining = storedTrainingInfoMap.get(TrainingManager.HIGHEST_NUMBER_OF_VOICES).intValue();
////		boolean isTraining = false;
////		boolean argModelDuration = false;
////		if (!isBidirectional) {
////			if (modelDuration) {
////				argModelDuration = true;
////			}
////		}
////		if (isBidirectional) {
////			if (modelDuration && modelDurationAgain) {
////				argModelDuration = true;
////			}
////		}
////		String errorSpecifications = errorCalculator.getErrorSpecifications(assignmentErrors, allPredictedVoices,
////			groundTruthVoiceLabels,	equalDurationUnisonsInfo, highestNumberOfVoicesTraining, isTraining, argModelDuration);
////		double smallestNetworkError = -1;
////		String errorSpecifications = errorCalculator.getErrorSpecifications(modelParameters,
////			assignmentErrors, smallestNetworkError, allPredictedVoices,	argGroundTruthVoiceLabels,
////			getEqualDurationUnisonsInfo(), false/*, argModelDuration*/);
//		results = results.concat(errorSpecifications);
//
//		// 4. Get label and network output details and add to results 		
//		String details = 
//			getActualAndPredictedVoicesDetails(modelParameters, /*bestWeights,*/ 
//			conflictIndices, allPredictedVoices, allPredictedDurationLabels, 
//			allMetricPositions,	/*argVoicesCoDNotes,*/ backwardsMapping);
////			chordSizes, allNetworkOutputs, groundTruthVoiceLabels, equalDurationUnisonsInfo, 
////			groundTruthDurationLabels, voicesCoDNotes, groundTruthChordVoiceLabels, 
////			possibleVoiceAssignmentsAllChords, allBestVoiceAssignments, allHighestNetworkOutputs);
//		results = results.concat(details);
//		
//		// 5. Only in the application case, and only for N2N: add the conflictsRecord to results
//		if (learningApproach == NOTE_CLASS) {
//			results = results.concat(additionalInfo);
//		}
//
//		// 6. Only if bidirectional: add the bidirectional conflicts record to results
////		if (isBidirectional) {
//		boolean argIsBidirectional = 
//			ToolBox.toBoolean(modelParameters.get(TrainingManager.IS_BIDIRECTIONAL).intValue());
//		if (argIsBidirectional) {
//			results = results.concat("CONFLICTS\r\n" + conflictsRecordTest);
//		}
//
////		System.out.println(errorSpecifications);
//		
//		// 7. Make the header, containing the test end time, and add results to it
//		String header = "";
//		header = header.concat("===================================" + "\r\n");
//		header = header.concat("Test started   " + startTime + "\r\n");   
//		header = header.concat("Test completed " + ToolBox.getTimeStamp() + "\r\n");
//		header = header.concat("===================================" + "\r\n" + "\r\n");
//		results = header.concat(results);
//		
//		return results;
		return null;
	}


	/**
	* Returns a String containing (1) for each note or chord, the ground truth label(s) + the actual voice(s) and
	* the network output + the predicted voice(s). In note-to-note application mode, to this are added (2) conflict
	* reassignments; (3) conflicts; (4) application process; also, any adaptations made to predicted voices due to 
	* conflicts are added in step (1).
	*  
	* @param argTrainingInfoMap
	* @param bestWeights
	* @param conflictIndices In bwd order when using bwd model
	* @param predictedVoices Application case only; in bwd order when using bwd model
	* @param predictedDurationLabels Application case only; in fwd order when using bwd model
	* @param allMetricPositions Is <code>null</code> in training mode; in fwd order in test/application mode
	* @param backwardsMapping Is <code>null</code> in training mode and when using fwd model in test/application mode
	* @return
	*/
	private String getActualAndPredictedVoicesDetailsZOALSHETWAS(Map<String, Double> argTrainingInfoMap,
		List<Double> bestWeights, List<List<Integer>> conflictIndices, List<List<Integer>> 
		predictedVoices, List<List<Double>> predictedDurationLabels, /*List<Rational[]> 
		allMetricPositions,*/ List<Integer> backwardsMapping) { 

		String labelAndNetworkOutputDetails = "";
//		int learningApproach = argTrainingInfoMap.get(MODELLING_APPROACH).intValue();
////		boolean errorMeasurementIsStrict = false;
////		if (argTrainingInfoMap.get(ERROR_MEASUREMENT).intValue() == STRICT) {
////			errorMeasurementIsStrict = true;
////		}
//
//		// Initialise the network with the given weights. When training, these are the best weights 
//		// from the current training run; when testing, they are the best weights from the training 
//		initWeightsFromList(bestWeights);
//		
//		// Make a DecimalFormat for formatting long double values to four decimal places
//		DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.UK);
//		otherSymbols.setDecimalSeparator('.'); 
//		DecimalFormat decForm = new DecimalFormat("0.0000", otherSymbols);
//
//		// Write out labels and network output details for each note
//	 	labelAndNetworkOutputDetails = detailsRec + "\r\n";
//		if (learningApproach == NOTE_CLASS) {
////			double allowCoD = argTrainingInfoMap.get(ALLOW_COD);
//			boolean allowCoD = ToolBox.toBoolean(argTrainingInfoMap.get(ALLOW_COD).intValue());
//			double deviationThreshold = argTrainingInfoMap.get(DEVIATION_THRESHOLD);
//			List<Integer> indicesOfCorrectToCorrectVoices = new ArrayList<Integer>();
//			List<Integer> indicesOfCorrectToIncorrectVoices = new ArrayList<Integer>();
//			List<Integer> indicesOfIncorrectToCorrectVoices = new ArrayList<Integer>();
//			List<Integer> indicesOfIncorrectToIncorrectVoices = new ArrayList<Integer>();
//
//			List<Integer> indicesOfCorrectToCorrectDurations = new ArrayList<Integer>();
//			List<Integer> indicesOfCorrectToIncorrectDurations = new ArrayList<Integer>();
//			List<Integer> indicesOfIncorrectToCorrectDurations = new ArrayList<Integer>();
//			List<Integer> indicesOfIncorrectToIncorrectDurations = new ArrayList<Integer>();
//
//			// For each note
//			int chordIndex = 0;
////			if (backwardsMapping != null) {
////				chordIndex = chordSizes.size();
////			}
//			int notesLeftInChord = chordSizes.get(chordIndex);
////			boolean durationModelled = false;
//			for (int i = 0; i < noteFeatures.size(); i++) {	
//				// There are two indices: 
//				// (1) i is used when the list the element is taken from is ordered in the same manner as noteFeatures (when 
//				//     using the fwd model, this goes for all lists). The lists ordered in the same manner as noteFeatures
//				//     are groundTruthVoiceLabels, groundTruthDurationLabels, voicesCoDNotes, and equalDurationUnisonsInfo
//				// (2) noteIndex is used when the list the element is taken from is always ordered as when using the fwd model
//				//     (when using the fwd model, noteIndex will thus always be the same as i)
//				//     Lists indexed with noteIndex can only be used in test/application mode, as in training mode multiple 
//				//     occurrences of noteIndex may occur
//				int noteIndex = i;
//				if (backwardsMapping != null) {
//					noteIndex = backwardsMapping.get(i);
//				}
//
//				double[] currentOutput = allNetworkOutputs.get(i); // HIER OK
//				double[] currentVoicesOutput = Arrays.copyOfRange(currentOutput, 0, MAX_NUM_VOICES_N2N); // needed to make predictedVoiceLabelAsStringArray
//				double[] currentDurationsOutput = null; // needed to make predictedDurationLabelAsStringArray
////				boolean durationModelled = false;
//				if (currentOutput.length > MAX_NUM_VOICES_N2N) {
//					modelDuration = true;
////					durationModelled = true;
//					currentDurationsOutput = Arrays.copyOfRange(currentOutput, MAX_NUM_VOICES_N2N, currentOutput.length);
//				}
//
//				// 1. Determine the note- and chord indices
//				String noteAndChordIndices = "noteIndex = " + i + " (chordIndex = " + chordIndex + ")"; // HIER OK
//				if (allMetricPositions == null) {
//					noteAndChordIndices = noteAndChordIndices.concat("\r\n");
//				}
//				else {
//					noteAndChordIndices = noteAndChordIndices.concat(", bar " + 
//				    ToolBox.getMetricPositionAsString(allMetricPositions.get(noteIndex)) + "\r\n"); // HIER OK
//				}
//
//				// 2. Format voice information
//				String actualAndPredictedVoices = "";
//				// Determine ground truth voice label and voice(s) and add to actualAndPredictedVoices
//				List<Double> currentActualVoiceLabel = groundTruthVoiceLabels.get(i); // HIER OK
//				List<Integer> currentActualVoices = dataConverter.convertIntoListOfVoices(currentActualVoiceLabel);
//				String av = "voice  ";
//				if (currentActualVoices.size() == 2) {
//					av = "voices ";
//				}
////				actualAndPredictedVoices = actualAndPredictedVoices.concat("ground truth voice label" + "\r\n" + 
////					"  " + currentActualVoiceLabel + "\t\t\t\t\t\t" + " --> voice(s) " + currentActualVoices + "\r\n");
//				actualAndPredictedVoices = actualAndPredictedVoices.concat("ground truth voice label" + "\r\n" + 
//					"  " + currentActualVoiceLabel + "\t\t\t\t\t\t" + " --> " + av + currentActualVoices + "\r\n");
//				// Determine predicted voice label and voice(s) and add to actualAndPredictedVoices
//				// Predicted voice label
//				String[] predictedVoiceLabelAsStringArray = new String[currentVoicesOutput.length]; 
//				for (int j = 0; j < currentVoicesOutput.length; j++) {
//					String df = decForm.format(currentVoicesOutput[j]);
//					predictedVoiceLabelAsStringArray[j] = df;
//				}		    
//				// Predicted voice(s)
//				String predictedVoicesString = "";
//				String pv = "voice  ";
//
//				// a. If there is a conflict to report (only in the application case): the initially predicted voice(s) must be 
//				// calculated as the argument predictedVoices gives the adapted voice(s) as they are after any conflicts 
//				// have been resolved
//				if (conflictIndices != null && conflictIndices.get(0).contains(i)) { // HIER OK		 	
//					// Determine the voice(s) predicted initially and the adapted voice(s) and add to predictedVoicesString
//					List<Integer> voicesPredictedInitially =  
//						outputEvaluator.interpretNetworkOutput(currentOutput, allowCoD,	deviationThreshold).get(0);		 	  	
//					List<Integer> predictedVoicesAdapted = predictedVoices.get(i); // HIER OK
//					if (predictedVoicesAdapted.size() == 2) {
//						pv = "voices ";
//					}
//					predictedVoicesString = voicesPredictedInitially + " --> adapted to " + predictedVoicesAdapted 
//						+ " because of conflict (see CONFLICTS)";  	  	
//					// Keep track of the conflict reassignments
//					boolean voicesPredictedInitiallyCorrectly = false;
//					boolean predictedVoicesAdaptedCorrectly = false;
//					boolean voicesPredictedInitiallyCorrectlyUpper = false;
//					boolean predictedVoicesAdaptedCorrectlyUpper = false;
//					// a. In the tablature case, where equal duration unisons do not apply
//					if (equalDurationUnisonsInfo == null) { 	
//						voicesPredictedInitiallyCorrectly = 
//							errorCalculator.assertCorrectness(/*errorMeasurementIsStrict,*/ 
//							voicesPredictedInitially, currentActualVoices);
//						predictedVoicesAdaptedCorrectly = 
//							errorCalculator.assertCorrectness(/*errorMeasurementIsStrict,*/ 
//							predictedVoicesAdapted, currentActualVoices);
//					}
//					// b. In the non-tablature case, when the note at index i is not part of an EDU
//					else if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) == null) { // HIER OK
//						voicesPredictedInitiallyCorrectly = 
//							errorCalculator.assertCorrectness(/*errorMeasurementIsStrict,*/ 
//							voicesPredictedInitially, currentActualVoices);
//						predictedVoicesAdaptedCorrectly = 
//							errorCalculator.assertCorrectness(/*errorMeasurementIsStrict,*/ 
//							predictedVoicesAdapted, currentActualVoices);
//					}
//					// c. In the non-tablature case, when the note at index i is part of an EDU
//					else if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) != null) { // HIER OK
//						// Only if the note at index i is not the last note
//						if (i != noteFeatures.size() - 1) {
//							// Determine the predicted and adapted voices for the lower and upper EDUnotes (.get(0) can be used
//							// because in the non-tablature case there are no CoDs and the lists will contain only one element)
//							int voicePredInitiallyLowerNote = voicesPredictedInitially.get(0);
//							double[] outputNextNote = allNetworkOutputs.get(i + 1); // HIER OK
//							int voicePredInitiallyUpperNote = outputEvaluator.interpretNetworkOutput(outputNextNote, allowCoD,
//							  	deviationThreshold).get(0).get(0);
//							int adaptedVoiceLowerNote = predictedVoicesAdapted.get(0);
//							int adaptedVoiceUpperNote = predictedVoices.get(i + 1).get(0); // HIER OK
//							// Determine for both EDUnotes whether the predicted and adapted voices are correct
//							List<Integer[]> predictedAndAdaptedVoices = new ArrayList<Integer[]>();
//							predictedAndAdaptedVoices.add(new Integer[]{voicePredInitiallyLowerNote, voicePredInitiallyUpperNote});
//							predictedAndAdaptedVoices.add(new Integer[]{adaptedVoiceLowerNote, adaptedVoiceUpperNote});
//							List<Integer> allowedVoices = Arrays.asList(new Integer[]{equalDurationUnisonsInfo.get(i)[1], 
//								equalDurationUnisonsInfo.get(i)[0]}); // HIER OK
//							boolean[][] unisonNotesPredictedCorrectly = 
//								errorCalculator.assertCorrectnessEDUNotes(/*errorMeasurementIsStrict,*/
//								predictedAndAdaptedVoices, allowedVoices); 
//							boolean lowerUnisonNotePredictedCorrectly = unisonNotesPredictedCorrectly[0][0];
//							boolean upperUnisonNotePredictedCorrectly = unisonNotesPredictedCorrectly[0][1];
//							boolean lowerUnisonNoteAdaptedCorrectly = unisonNotesPredictedCorrectly[1][0];
//							boolean upperUnisonNoteAdaptedCorrectly = unisonNotesPredictedCorrectly[1][1];
//							// Set variables
//							voicesPredictedInitiallyCorrectly = lowerUnisonNotePredictedCorrectly;
//							predictedVoicesAdaptedCorrectly = lowerUnisonNoteAdaptedCorrectly;
//							voicesPredictedInitiallyCorrectlyUpper = upperUnisonNotePredictedCorrectly;
//							predictedVoicesAdaptedCorrectlyUpper = upperUnisonNoteAdaptedCorrectly;
//						}
//					}
//					// Determine correctness of reassignments
//					// 1. In the tablature case, and in the non-tablature case for non EDUnotes and lower EDUnotes
//					// a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
//					if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == true) {
//						indicesOfCorrectToCorrectVoices.add(i); // HIER OK
//					}
//					// b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
//					if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == false) {
//						indicesOfCorrectToIncorrectVoices.add(i); // HIER OK
//					}
//					// c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
//					if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == true) {
//						indicesOfIncorrectToCorrectVoices.add(i); // HIER OK
//					}
//					// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
//					if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == false) {
//						indicesOfIncorrectToIncorrectVoices.add(i); // HIER OK
//					}  	
//					// 2. In the non-tablature case for upper EDUnotes 
//					// NB: To avoid the index of the upper EDUnote being added twice (once when i is the index of the lower 
//					// EDUnote and once that of the upper), it must only be added when i is the index of the lower EDUnote
//					// TODO How is this implemented -- by means of the i + 1?
//					if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) != null) { // HIER OK
//						// a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
//						if (voicesPredictedInitiallyCorrectlyUpper == true && predictedVoicesAdaptedCorrectlyUpper == true) {
//							if (!indicesOfCorrectToCorrectVoices.contains(i + 1)) { // HIER OK
//								indicesOfCorrectToCorrectVoices.add(i + 1); // HIER OK
//							}
//						}
//						// b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
//						if (voicesPredictedInitiallyCorrectlyUpper == true && predictedVoicesAdaptedCorrectlyUpper == false) {
//							if (!indicesOfCorrectToIncorrectVoices.contains(i + 1)) { // HIER OK
//								indicesOfCorrectToIncorrectVoices.add(i + 1); // HIER OK
//							}
//						}
//						// c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
//						if (voicesPredictedInitiallyCorrectlyUpper == false && predictedVoicesAdaptedCorrectlyUpper == true) {
//							if (!indicesOfIncorrectToCorrectVoices.contains(i + 1)) { // HIER OK
//								indicesOfIncorrectToCorrectVoices.add(i + 1); // HIER OK
//							}
//						}
//						// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
//						if (voicesPredictedInitiallyCorrectlyUpper == false && predictedVoicesAdaptedCorrectlyUpper == false) {
//							if (!indicesOfIncorrectToIncorrectVoices.contains(i + 1)) { // HIER OK
//								indicesOfIncorrectToIncorrectVoices.add(i + 1); // HIER OK
//							}
//						}
//					}
//				}
//				// b. In training or test mode, or in application mode if there is no conflict to report
//				else {
//					List<Integer> predVoices = null;
//					// In training or test mode, the predicted voices can be deduced directly from currentOutput as there are 
//					// no adaptations due to conflicts
//					if (conflictIndices == null) {	
//						List<Integer> voicesPredictedInitially = // TODO this is also above; combine duplicate and simplify 
//							outputEvaluator.interpretNetworkOutput(currentOutput, allowCoD,	deviationThreshold).get(0);
//						predVoices = voicesPredictedInitially;
//					}
//					// In application mode, the predicted voices are taken from the argument predictedVoices
//					else {
//						List<Integer> predictedVoicesAdapted = predictedVoices.get(i); // HIER OK
//						predVoices = predictedVoicesAdapted;
//					}
//					predictedVoicesString = "" + predVoices;
//					if (predVoices.size() == 2) {
//						pv = "voices ";
//					}
//				}
//				// Add to actualAndPredictedVoices
////				actualAndPredictedVoices = actualAndPredictedVoices.concat("network output for voice" + "\r\n" + 
////					"  " + Arrays.toString(predictedVoiceLabelAsStringArray) +  "\t\t\t\t" + " --> voice(s) " + 
////					predictedVoicesString + "\r\n");
//				actualAndPredictedVoices = actualAndPredictedVoices.concat("network output for voice" + "\r\n" + 
//					"  " + Arrays.toString(predictedVoiceLabelAsStringArray) +  "\t\t\t\t" + " --> " + pv + 
//					predictedVoicesString + "\r\n");
//
//				// 3. If duration is modelled: format duration information
//				String actualAndPredictedDurations = "";
//				if (!isBidirectional && modelDuration || isBidirectional && modelDuration && modelDurationAgain) {
////				if (modelDuration) {
//					// Determine ground truth duration label and duration(s) and add to actualAndPredictedDurations
//					List<Double> currentActualDurationLabel = groundTruthDurationLabels.get(i); // HIER OK
//					Rational[] currentActualDurations = dataConverter.convertIntoDuration(currentActualDurationLabel);
//					for (Rational r : currentActualDurations) {
//						r.reduce();
//					}
//					String actualDurationLabelBroken = 
//						ToolBox.breakString(currentActualDurationLabel.toString(), 8, ',', "  ");
//					actualAndPredictedDurations = actualAndPredictedDurations.concat("ground truth duration label" + "\r\n" +
//						"  " + actualDurationLabelBroken + "\t\t\t\t" + " --> duration(s) "); // + Arrays.asList(currentActualDurations));
//					// In the case of a CoD with two durations: also add the durations (in the correct order)
//					if (currentActualVoices.size() == 2 && currentActualDurations.length == 2) { 
////						// currentVoicesCoDNotes is ordered with the voice of the longer CoDnote first
////						Integer[] currentVoicesCoDNotes = voicesCoDNotes.get(noteIndex);
////						int voiceOfLonger = currentVoicesCoDNotes[0];    	
////						// currentActualDurations is ordered with the longer duration first
////						Rational longerDur = currentActualDurations[0];
////						Rational shorterDur = currentActualDurations[1];
//						// Turn currentActualDurations into a List
//						List<Rational> currentActualDurationsAsList = Arrays.asList(new Rational[]{currentActualDurations[0],
//							currentActualDurations[1]});
//						// currentActualVoices is ordered with the higher voice first; currentActualDurations(AsList) with the 
//						// longer duration first. Thus, if the higher voice has the longer duration: currentActualDurationsAsList
//						// is in the right order; if not: its elements must be swapped
//						Integer[] currentVoicesCoDNotes = voicesCoDNotes.get(i);
//						// Get the voice of the shorter CoDnote. currentVoicesCoDNotes is ordered with the voice of the longer first
//						int voiceOfShorter = currentVoicesCoDNotes[1];
//						if (currentActualVoices.get(0) == voiceOfShorter) {
//							Collections.reverse(currentActualDurationsAsList);
//						}
//						actualAndPredictedDurations = actualAndPredictedDurations.concat(currentActualDurationsAsList + 
//							" (for voices " + currentActualVoices + ")" + "\r\n");
////						actualAndPredictedDurations = actualAndPredictedDurations.concat(" (longer CoDnote first)");
//					}
//					// In the case of a CoD with one duration or no CoD
//					else {
//						actualAndPredictedDurations = actualAndPredictedDurations.concat(Arrays.asList(currentActualDurations) + "\r\n");
//					}
//
//					// Determine predicted duration label and duration(s) and add to actualAndPredictedDurations
//					// Predicted duration label
//					String[] predictedDurationLabelAsStringArray = new String[currentDurationsOutput.length]; 
//					for (int j = 0; j < currentDurationsOutput.length; j++) {
//						String df = decForm.format(currentDurationsOutput[j]);
//						predictedDurationLabelAsStringArray[j] = df;
//					}
//
////					List<Integer> predictedDurationsAsList = // SAME
////						outputEvaluator.interpretNetworkOutput(currentOutput, allowCoD,	deviationThreshold).get(1);
////					List<Double> durationLabelPredicted = dataConverter.convertIntoDurationLabel(predictedDurationsAsList);
////					Rational[] predictedDurations = dataConverter.convertIntoDuration(durationLabelPredicted); 
////					for (Rational r : predictedDurations) {
////						r.reduce();
////					}
////					actualAndPredictedDurations = actualAndPredictedDurations.concat("network output for duration" + "\r\n" +
////						"  " + Arrays.toString(predictedDurationLabelAsStringArray) + " --> duration(s) " + Arrays.asList(predictedDurations) + "\r\n");
//
//			    // Predicted duration(s)
//			 	  String predictedDurationsString = "";
//			    // a. If there is a conflict to report (only in the application case): the initially predicted duration must
//			 	  // be calculated as the argument predictedDurationLabels gives the adapted duration as it is after any conflicts 
//			 	  // have been resolved
//			 	  if (conflictIndices != null && conflictIndices.get(1).contains(i)) { // HIER OK		 	
//			 	  	// Determine the duration(s) predicted initially and the adapted duration(s) and add to predictedDurationsString
//			 	    // NB: durationsPredictedInitially and predictedDurationsAdapted will have only one element as currently, 
//			 	  	// only one duration per note is predicted TODO
//			 	  	List<Integer> durationsPredictedInitiallyAsList = //SAME 
//			 	  		outputEvaluator.interpretNetworkOutput(currentOutput, allowCoD,	deviationThreshold).get(1);
//			 	  	List<Double> durationLabelPredictedInitially = 
//			 	  		dataConverter.convertIntoDurationLabel(durationsPredictedInitiallyAsList);
//			 	  	Rational[] durationsPredictedInitially = dataConverter.convertIntoDuration(durationLabelPredictedInitially);
//			 	  	for (Rational r : durationsPredictedInitially) {
//				     	r.reduce();
//				    }
//			 	  	List<Double> predictedDurationLabelAdapted = predictedDurationLabels.get(noteIndex); // HIER OK
//			 	  	Rational[] predictedDurationsAdapted = dataConverter.convertIntoDuration(predictedDurationLabelAdapted); 
//				 	  for (Rational r : predictedDurationsAdapted) {
//				     	r.reduce();
//				    }
//			 	  	predictedDurationsString = Arrays.toString(durationsPredictedInitially) + " --> adapted to " + 
//				      Arrays.toString(predictedDurationsAdapted) + " because of conflict (see CONFLICTS)";
//			 	  	
//			 	    // Keep track of the conflict reassignments
//			 	  	boolean durationsPredictedInitiallyCorrectlyOrHalfCorrectly = false;
//			 	  	boolean predictedDurationsAdaptedCorrectlyOrHalfCorrectly = false;
//			 	  	// a. If currentActualDurations contains one element (i.e., if it goes with a note that is not a CoD, or with
//			 	  	// a note that is a CoD whose notes have the same duration)
//			 	  	if (currentActualDurations.length == 1) {
//			 	  		// Durations predicted initially
//			 	  		if (durationsPredictedInitially[0].equals(currentActualDurations[0])) {
//			 	  			durationsPredictedInitiallyCorrectlyOrHalfCorrectly = true;
//		 	  			}
//			 	  		// Adapted durations
//			 	  		if (predictedDurationsAdapted[0].equals(currentActualDurations[0])) {
//			 	  			predictedDurationsAdaptedCorrectlyOrHalfCorrectly = true;
//		 	  			}
//			 	  	}
//			 	    // b. If currentActualDurations contains two elements (i.e., if it goes with a note that is CoD whose 
//			 	  	// notes have different durations) 
//			 	  	if (currentActualDurations.length == 2) {
//		 	  			// Durations predicted initially
//			 	  		// If the duration predicted initially is the same as one of the actual durations 
//		 	  			if (durationsPredictedInitially[0].equals(currentActualDurations[0]) || 
//		 	  				durationsPredictedInitially[0].equals(currentActualDurations[1])) {
//		 	  				durationsPredictedInitiallyCorrectlyOrHalfCorrectly = true;
//		 	  			}
//		 	  			// Adapted durations
//		 	  	    // If the adapted duration is the same as one of the actual durations
//		 	  			if (predictedDurationsAdapted[0].equals(currentActualDurations[0]) || 
//		 	  				predictedDurationsAdapted[0].equals(currentActualDurations[1])) {
//		 	  				predictedDurationsAdaptedCorrectlyOrHalfCorrectly = true;
//			 	  		}
//			 	  	}
//			 	  	
//			 	  	// Determine correctness of reassignments
//			 	    // a. If the duration(s) were initially predicted (half) correctly and were adapted (half) correctly
//			 	  	if (durationsPredictedInitiallyCorrectlyOrHalfCorrectly == true && 
//			 	  		predictedDurationsAdaptedCorrectlyOrHalfCorrectly == true) {
//	     		    indicesOfCorrectToCorrectDurations.add(i); // HIER OK
//	   	  	  }
//	     		  // b. If the duration(s) were initially predicted (half) correctly but were adapted incorrectly 
//	     		  if (durationsPredictedInitiallyCorrectlyOrHalfCorrectly == true && 
//	     		  	predictedDurationsAdaptedCorrectlyOrHalfCorrectly == false) {
//	      	  	indicesOfCorrectToIncorrectDurations.add(i); // HIER OK
//	      	  }
//	   	  	  // c. If the duration(s) were initially predicted incorrectly but were adapted (half) correctly 
//	    		  if (durationsPredictedInitiallyCorrectlyOrHalfCorrectly == false && 
//	    		  	predictedDurationsAdaptedCorrectlyOrHalfCorrectly == true) {
//	      	    indicesOfIncorrectToCorrectDurations.add(i); // HIER OK
//	        	}
//	   	    	// d. If the duration(s) were initially predicted incorrectly and were adapted incorrectly
//	     	  	if (durationsPredictedInitiallyCorrectlyOrHalfCorrectly == false && 
//	     	  		predictedDurationsAdaptedCorrectlyOrHalfCorrectly == false) {
//	        		indicesOfIncorrectToIncorrectDurations.add(i); // HIER OK
//	        	}
//			 	  	
//			 	  	
//			 	  	
//			 	    // ########## VERVANGEN			 	  	
////			 	  	// a. In the tablature case, where equal duration unisons do not apply
////			 	  	if (equalDurationUnisonsInfo == null) { 	
////	     		    voicesPredictedInitiallyCorrectly = errorCalculator.assertCorrectness(errorMeasurementIsStrict, 
////	     		  	  voicesPredictedInitially,	actualVoices);
////	     		    predictedVoicesAdaptedCorrectly = errorCalculator.assertCorrectness(errorMeasurementIsStrict, 
////	     		  	  predictedVoicesAdapted, actualVoices);
////			 	  	}
//			 	  	
////			 	  	// Determine correctness of reassignments
////			 	  	// 1. In the tablature case, and in the non-tablature case for non EDUnotes and lower EDUnotes
////	   		    // a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
////	   		    if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == true) {
////	     		    indicesOfCorrectToCorrect.add(i); // HIER OK
////	   	  	  }
////	     		  // b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
////	     		  if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == false) {
////	      	  	indicesOfCorrectToIncorrect.add(i); // HIER OK
////	      	  }
////	   	  	  // c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
////	    		  if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == true) {
////	      	    indicesOfIncorrectToCorrect.add(i); // HIER OK
////	        	}
////	   	    	// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
////	     	  	if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == false) {
////	        		indicesOfIncorrectToIncorrect.add(i); // HIER OK
////	        	}  	
//	     	  	
//			 	  	
////			 	    // b. In the non-tablature case, when the note at index i is not part of an EDU
////			 	  	else if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) == null) { // HIER OK
////			 	  		voicesPredictedInitiallyCorrectly = errorCalculator.assertCorrectness(errorMeasurementIsStrict, 
////		     		  	voicesPredictedInitially,	actualVoices);
////		     		  predictedVoicesAdaptedCorrectly = errorCalculator.assertCorrectness(errorMeasurementIsStrict, 
////		     		    predictedVoicesAdapted, actualVoices);
////			 	  	}
////			 	    // c. In the non-tablature case, when the note at index i is part of an EDU
////			 	  	else if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) != null) { // HIER OK
////			 	  		// Only if the note at index i is not the last note
////			 	  		if (i != noteFeatures.size() - 1) {
////					 	    // Determine the predicted and adapted voices for the lower and upper EDUnotes (.get(0) can be used
////			 	  		  // because in the non-tablature case there are no CoDs and the lists will contain only one element)
////	  		 	  		int voicePredInitiallyLowerNote = voicesPredictedInitially.get(0);
////	  						double[] outputNextNote = allNetworkOutputs.get(i + 1); // HIER OK
////	  		 	  		int voicePredInitiallyUpperNote = outputEvaluator.interpretNetworkOutput(outputNextNote, allowCoD,
////	  				 	  	deviationThreshold).get(0).get(0);
////	  				 	  int adaptedVoiceLowerNote = predictedVoicesAdapted.get(0);
////	  				 	  int adaptedVoiceUpperNote = predictedVoices.get(i + 1).get(0); // HIER OK
////	  		 	  		// Determine for both EDUnotes whether the predicted and adapted voices are correct
////	  				 	  List<Integer[]> predictedAndAdaptedVoices = new ArrayList<Integer[]>();
////	  		 	      predictedAndAdaptedVoices.add(new Integer[]{voicePredInitiallyLowerNote, voicePredInitiallyUpperNote});
////	  		 	      predictedAndAdaptedVoices.add(new Integer[]{adaptedVoiceLowerNote, adaptedVoiceUpperNote});
////	  		 	      List<Integer> allowedVoices = Arrays.asList(new Integer[]{equalDurationUnisonsInfo.get(i)[1], 
////	  		 	      	equalDurationUnisonsInfo.get(i)[0]}); // HIER OK
////	  		 	      boolean[][] unisonNotesPredictedCorrectly = errorCalculator.assertCorrectnessEDUNotes(errorMeasurementIsStrict,
////	  		 	      	predictedAndAdaptedVoices, allowedVoices); 
////	  		 	      boolean lowerUnisonNotePredictedCorrectly = unisonNotesPredictedCorrectly[0][0];
////	  		 	      boolean upperUnisonNotePredictedCorrectly = unisonNotesPredictedCorrectly[0][1];
////	  		 	      boolean lowerUnisonNoteAdaptedCorrectly = unisonNotesPredictedCorrectly[1][0];
////	  		 	      boolean upperUnisonNoteAdaptedCorrectly = unisonNotesPredictedCorrectly[1][1];
////	  		 	      // Set variables
////	  		 	      voicesPredictedInitiallyCorrectly = lowerUnisonNotePredictedCorrectly;
////	  		 	      predictedVoicesAdaptedCorrectly = lowerUnisonNoteAdaptedCorrectly;
////	  		 	      voicesPredictedInitiallyCorrectlyUpper = upperUnisonNotePredictedCorrectly;
////	  		 	      predictedVoicesAdaptedCorrectlyUpper = upperUnisonNoteAdaptedCorrectly;
////			 	  		}
////			 	  	}
////			 	  	// Determine correctness of reassignments
////			 	  	// 1. In the tablature case, and in the non-tablature case for non EDUnotes and lower EDUnotes
////	   		    // a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
////	   		    if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == true) {
////	     		    indicesOfCorrectToCorrect.add(i); // HIER OK
////	   	  	  }
////	     		  // b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
////	     		  if (voicesPredictedInitiallyCorrectly == true && predictedVoicesAdaptedCorrectly == false) {
////	      	  	indicesOfCorrectToIncorrect.add(i); // HIER OK
////	      	  }
////	   	  	  // c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
////	    		  if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == true) {
////	      	    indicesOfIncorrectToCorrect.add(i); // HIER OK
////	        	}
////	   	    	// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
////	     	  	if (voicesPredictedInitiallyCorrectly == false && predictedVoicesAdaptedCorrectly == false) {
////	        		indicesOfIncorrectToIncorrect.add(i); // HIER OK
////	        	}  	
////	     	  	
////	     	  	// 2. In the non-tablature case for upper EDUnotes 
////	     	    // NB: To avoid the index of the upper EDUnote being added twice (once when i is the index of the lower 
////	     	  	// EDUnote and once that of the upper), it must only be added when i is the index of the lower EDUnote
////	     	  	// TODO How is this implemented -- by means of the i + 1?
////	     	  	if (equalDurationUnisonsInfo != null && equalDurationUnisonsInfo.get(i) != null) { // HIER OK
////	       	    // a. If the voice(s) were initially predicted correctly and were adapted into the correct voice(s)
////	       	  	if (voicesPredictedInitiallyCorrectlyUpper == true && predictedVoicesAdaptedCorrectlyUpper == true) {
////	       		    if (!indicesOfCorrectToCorrect.contains(i + 1)) { // HIER OK
////	     		    	  indicesOfCorrectToCorrect.add(i + 1); // HIER OK
////	       		    }
////	     	  	  }
////	       		  // b. If the voice(s) were initially predicted correctly but were adapted into the incorrect voice(s)
////	       	  	if (voicesPredictedInitiallyCorrectlyUpper == true && predictedVoicesAdaptedCorrectlyUpper == false) {
////	       		  	if (!indicesOfCorrectToIncorrect.contains(i + 1)) { // HIER OK
////	       		  	  indicesOfCorrectToIncorrect.add(i + 1); // HIER OK
////	       		  	}
////	        	  }
////	     	  	  // c. If the voice(s) were initially predicted incorrectly but were adapted into the correct voice(s) 
////	       	  	if (voicesPredictedInitiallyCorrectlyUpper == false && predictedVoicesAdaptedCorrectlyUpper == true) {
////	      		  	if (!indicesOfIncorrectToCorrect.contains(i + 1)) { // HIER OK
////	      		  	  indicesOfIncorrectToCorrect.add(i + 1); // HIER OK
////	      		  	}
////	          	}
////	     	    	// d. If the voice(s) were initially predicted incorrectly and were adapted into the incorrect voice(s)
////	       	  	if (voicesPredictedInitiallyCorrectlyUpper == false && predictedVoicesAdaptedCorrectlyUpper == false) {
////	       	  		if (!indicesOfIncorrectToIncorrect.contains(i + 1)) { // HIER OK
////	       	  		  indicesOfIncorrectToIncorrect.add(i + 1); // HIER OK
////	       	  		}
////	          	}
////	     	  	}
//	     	    // ########## VERVANGEN
//	     	  	
//	     	  }
//			    // b. In training or test mode, or in application mode if there is no conflict to report
//			 	  else { 
//			 	  	Rational[] predDurations = null;
//			 	  	// In training or test mode, the predicted durations can be deduced directly from currentOutput as there 
//			 	  	// are no adaptations due to conflicts
//			 	  	if (conflictIndices == null) {
//			 	  		List<Integer> durationsPredictedInitiallyAsList = // TODO this is also above; combine duplicate 
//					 	  	outputEvaluator.interpretNetworkOutput(currentOutput, allowCoD,	deviationThreshold).get(1);
//					 	  List<Double> durationLabelPredictedInitially = 
//					 	  	dataConverter.convertIntoDurationLabel(durationsPredictedInitiallyAsList);
//			 	  	  Rational[] durationsPredictedInitially = dataConverter.convertIntoDuration(durationLabelPredictedInitially);
//			 	  	  for (Rational r : durationsPredictedInitially) {
//				     	  r.reduce();
//				      }
//			 	  	  predDurations = durationsPredictedInitially;
//			 	  	}
//			 	    // In application mode, the predicted durations are taken from the argument predictedDurationLabels
//			 	  	else {
//			 	  	  List<Double> predictedDurationLabel = predictedDurationLabels.get(noteIndex); // HIER OK
//			 	  	  Rational[] predictedDuration = dataConverter.convertIntoDuration(predictedDurationLabel);
//			 	  	  predDurations = predictedDuration;
//			 	    }
//			 	  	predictedDurationsString = "" + Arrays.toString(predDurations);
//			 	  }
//			    // Add to actualAndPredictedDurations
//			 	  String predictedDurationLabelBroken = 
//			 	  	ToolBox.breakString(Arrays.toString(predictedDurationLabelAsStringArray), 8, ',' , "  ");
//			 	  actualAndPredictedDurations = actualAndPredictedDurations.concat("network output for duration" + "\r\n" + 
//				    "  " + predictedDurationLabelBroken +  "\t" + " --> duration(s) " + predictedDurationsString + "\r\n");
//		    }
//		     
//		 	  // 4. Combine the above into the labelAndNetworkOutputDetails for the note at index i 
//		 	  labelAndNetworkOutputDetails = labelAndNetworkOutputDetails.concat(noteAndChordIndices + 
//		 	  	actualAndPredictedVoices + actualAndPredictedDurations + "\r\n");
//		 	  		 	  
//		    // 5. For the next iteration of the for-loop:
//			  // Decrement notesLeftInChord 
//				notesLeftInChord--;
//				// Are there no more notes left in the chord? Increment chordIndex; reset notesLeftInchord
//				if (notesLeftInChord == 0) {
//					// Only if i is not the index of the last note
//					if (i + 1 != noteFeatures.size()) {
//						chordIndex++;
//						notesLeftInChord = chordSizes.get(chordIndex); 
//					}
//				} 	  
//			}
//			// Add information on conflict reassignments (only in the application case)
//			if (conflictIndices != null) {
//				labelAndNetworkOutputDetails = labelAndNetworkOutputDetails.concat("CONFLICT REASSIGNMENTS" + "\r\n");
//				if (conflictIndices.get(0).size() != 0) {
//  				labelAndNetworkOutputDetails = labelAndNetworkOutputDetails.concat(
//  			    "number of notes initially predicted correctly and reassigned correctly: " +
//  				  indicesOfCorrectToCorrectVoices.size() + "\r\n" +
//  				  "  at indices " + indicesOfCorrectToCorrectVoices + "\r\n" +
//  			    "number of notes initially predicted correctly and reassigned incorrectly: " + 
//  			    indicesOfCorrectToIncorrectVoices.size() + "\r\n" + 
//  			    "  at indices " + indicesOfCorrectToIncorrectVoices + "\r\n" + 
//  			    "number of notes initially predicted incorrectly and reassigned correctly: " +
//  			    indicesOfIncorrectToCorrectVoices.size() + "\r\n" +
//  			    "  at indices " + indicesOfIncorrectToCorrectVoices + "\r\n" +
//  			    "number of notes initially predicted incorrectly and reassigned incorrectly: " + 
//  			    indicesOfIncorrectToIncorrectVoices.size() + "\r\n" + 
//  			    "  at indices " + indicesOfIncorrectToIncorrectVoices + "\r\n");
//				}
//				if (modelDuration) {
//					if (conflictIndices.get(1).size() != 0) {
//						labelAndNetworkOutputDetails = labelAndNetworkOutputDetails.concat(
//	  			    "number of durations initially predicted correctly and reassigned correctly: " +
//	  				  indicesOfCorrectToCorrectDurations.size() + "\r\n" +
//	  				  "  at indices " + indicesOfCorrectToCorrectDurations + "\r\n" +
//	  			    "number of durations initially predicted correctly and reassigned incorrectly: " + 
//	  			    indicesOfCorrectToIncorrectDurations.size() + "\r\n" + 
//	  			    "  at indices " + indicesOfCorrectToIncorrectDurations + "\r\n" + 
//	  			    "number of durations initially predicted incorrectly and reassigned correctly: " +
//	  			    indicesOfIncorrectToCorrectDurations.size() + "\r\n" +
//	  			    "  at indices " + indicesOfIncorrectToCorrectDurations + "\r\n" +
//	  			    "number of durations initially predicted incorrectly and reassigned incorrectly: " + 
//	  			    indicesOfIncorrectToIncorrectDurations.size() + "\r\n" + 
//	  			    "  at indices " + indicesOfIncorrectToIncorrectDurations + "\r\n");
//					}
//				}
//				labelAndNetworkOutputDetails = labelAndNetworkOutputDetails.concat("\r\n");
//			}
//		}
//		else if (learningApproach == CHORD_REGRESS) {
//			int lowestNoteIndex = 0;
//		 	for (int i = 0; i < chordFeatures.size(); i++) {		   	
//		 		// Determine the chord- and note indices
//		 		List<List<Double>> currentChordLabels = groundTruthChordVoiceLabels.get(i); 
//		 		int chordSize = currentChordLabels.size();
//		 		String noteIndices = null;
//		 		if (chordSize == 1) {
//		 			noteIndices = " (noteIndex = " + lowestNoteIndex + ")"; 
//		 		}
//		 		else {
//		 			noteIndices = " (noteIndices = " + lowestNoteIndex + "-" + (lowestNoteIndex + chordSize - 1) + ")";
//		 		}
//        String chordAndNoteIndices = "chordIndex = " + i + noteIndices; // + "\r\n";
//				if (allMetricPositions == null) {
//					chordAndNoteIndices = chordAndNoteIndices.concat("\r\n");
//				}
//				else {
//					chordAndNoteIndices = chordAndNoteIndices.concat(", bar " + 
//						ToolBox.getMetricPositionAsString(allMetricPositions.get(lowestNoteIndex)) + "\r\n");
//				}
//				lowestNoteIndex += chordSize;
//				
//		  	// Format labels
//        String labelHeader = "ground truth voice labels" + "\r\n";
//		 		String chordLabelString = "";
//	      for (int j = 0; j < currentChordLabels.size(); j++) {
//	      	List<Double> currentNoteLabel = currentChordLabels.get(j);
//	        List<Integer> actualVoices = dataConverter.convertIntoListOfVoices(currentNoteLabel);
//	        String av = "voice  ";
//	        if (actualVoices.size() == 2) {
//	        	av = "voices ";
//	        }
////	        chordLabelString = chordLabelString.concat(currentNoteLabel + "\t" + "\t" + "\t" + "\t" + 
////            " --> voice(s) " + actualVoices + "\r\n");
//	        chordLabelString = chordLabelString.concat(currentNoteLabel + "\t" + "\t" + "\t" + "\t" + 
//	            " --> " + av + actualVoices + "\r\n");
//	      }
//	      
//		  	// Format output
//	      String outputHeader = "highest network output" + "\r\n";
//		  	List<List<Integer>> possibleVoiceAssignmentsCurrentChord = possibleVoiceAssignmentsAllChords.get(i);
//		  	List<Integer> bestVoiceAssignment = allBestVoiceAssignments.get(i);
//		  	int bestEval = possibleVoiceAssignmentsCurrentChord.indexOf(bestVoiceAssignment);
//		  	List<List<Double>> chordLabels = dataConverter.getChordVoiceLabels(bestVoiceAssignment);
//		 		List<List<Integer>> predictedChordVoices = dataConverter.getVoicesInChord(chordLabels);
//		  	double highestNetworkOutput = allHighestNetworkOutputs.get(i);  	
//		  	String outputString = "[" + decForm.format(highestNetworkOutput) + "] for voice assignment no. " +
//		  	  bestEval + ": " + bestVoiceAssignment + /*"\t" +*/ " --> voice(s) " + predictedChordVoices + "\r\n" + "\r\n"; // hier
//		  	
//		  	labelAndNetworkOutputDetails = labelAndNetworkOutputDetails.concat(chordAndNoteIndices + labelHeader +
//		  		chordLabelString + outputHeader +	outputString);
//		  }
//		}
		return labelAndNetworkOutputDetails;
	}
	
	
//	/**
//	 * Creates a single training example out of the given features and label. The training example is wrapped in
//	 * an MLDataPair.
//	 * 
//	 * @param argFeatures
//	 * @param argLabels
//	 */
//	private void createTrainingExampleNOTINUSE(List<Double> argFeatures, List<Double> argLabels) {
//	  // Turn Lists into Arrays
//		double[] input = new double[argFeatures.size()];
//		for (int i = 0; i < argFeatures.size(); i++) {
//			input[i] = argFeatures.get(i);
//		}
//		dataPair.setInputArray(input);
//		
//		double[] output = new double[argLabels.size()];
//		for (int i = 0; i < argLabels.size(); i++) {
//			output[i] = argLabels.get(i);
//		}
//		dataPair.setIdealArray(output);
//	}
	
	
	/**
	 * Gets the voices the network predicts for noteFeatures or chordFeatures with the given settings.
	 * 
	 * @param learningApproach
	 * @param allowCoD
	 * @param deviationThreshold
	 * @param boolean isTrainingOrTestMode
	 * @param dataConverter
	 * @param argOutputEvaluator
	 * @param argErrorCalculator
	 * @return
	 */
	// FIXME commented out 12.11.2016
	private List<List<Integer>> determinePredictedVoicesNOTINUSE(int learningApproach, boolean allowCoD, 
		double deviationThreshold, boolean isTrainingOrTestMode) {
		List<List<Integer>> allPredictedVoices = new ArrayList<List<Integer>>();

//		if (learningApproach == NOTE_CLASS) { 
//      // allNetworkOutputs must be recreated every time this method is called in the training case. In the test
//    	// case, this method is only called once, so recreation is not strictly necessary -- but it is harmless nevertheless
//      allNetworkOutputs = new ArrayList<double[]>(); // NIEUW 7-1
//    	
//    	int numberOfNotes = noteFeatures.size();
//      for (int noteIndex = 0; noteIndex < numberOfNotes; noteIndex++) {
//        // Evaluate the network for the current onset
//        double[] predictedLabel = evalNetwork(noteFeatures.get(noteIndex));
//        allNetworkOutputs.add(predictedLabel); // NIEUW 7-1
//        // Interpret the output and get the predicted voice(s); add that List to allPredictedVoices
//        List<Integer> predictedVoices = 
//        	outputEvaluator.interpretNetworkOutput(predictedLabel, allowCoD, deviationThreshold).get(0);
//        allPredictedVoices.add(predictedVoices);
//      }
//    }
//    if (learningApproach == CHORD_REGRESS) {
//  		int numberOfChords = chordFeatures.size();
//  	  // allHighestNetworkOutputs, as well as allBestVoiceAssignments, must be recreated every time 
//  	  // this method is called in the training case. In the test case, this method is only called once, so
//  	  // recreation is not strictly necessary -- but it is harmless nevertheless
//      allHighestNetworkOutputs = new ArrayList<Double>(); 
//      allBestVoiceAssignments = new ArrayList<List<Integer>>();
//  		// For each chord
//      for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
////		    List<List<Integer>> currentChordOnsetProperties = allChordOnsetProperties.get(chordIndex);
////		    Integer[][] currentBasicTabSymbolPropertiesChord = allBasicTabSymbolPropertiesChord.get(chordIndex); 
//		    // a. For all possible feature vectors for this chord: evaluate the network and add the result, the
//		    // network output, to currentNetworkOutputs
//		    List<List<Double>> currentChordFeatures = chordFeatures.get(chordIndex);
//      	List<List<Integer>> currentPossibleVoiceAssignments = possibleVoiceAssignmentsAllChords.get(chordIndex);  	
//      	List<Double> currentNetworkOutputs = new ArrayList<Double>();
//  		  for (int j = 0; j < currentChordFeatures.size(); j++) {
//  		  	List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
//    	 		double[] currentNetworkOutput = evalNetwork(currentChordFeatureVector);
//    	 		currentNetworkOutputs.add(currentNetworkOutput[0]);
//  		  	if (Double.isNaN(currentNetworkOutput[0])) { // TODO remove
//  		  		System.out.println("Network output is NaN.");
//  		  		System.exit(0);
//  		  	}
//  		  }
//  		  // b. Add the highest network output to allHighestNetworkOutputs; it does not matter whether it
//  		  // appears more than once (if this happens, it is solved within determineBestVoiceAssignment() -- which
//  		  // is called in training, test, and application case)
//  		  double currentHighestNetworkOutput = Collections.max(currentNetworkOutputs);
////  		  if (Collections.frequency(currentNetworkOutputs, currentHighestNetworkOutput) > 1) {
////  		  	System.out.println("Highest network output appears more than once.");
////		  		System.exit(0);
////  		  }
//  		  allHighestNetworkOutputs.add(currentHighestNetworkOutput);
//  		  
//  		  // c. Determine the best voice assignment and add it to allBestVoiceAssignments. Because it is possible 
//  		  // that different voice assignments result in the same network output, the highest output may occur
//  		  // multiple times. If this is the case, less likely candidates must be filtered out; this happens 
//  		  // inside determineBestVoiceAssignment()
//  		  List<Integer> predictedBestVoiceAssignment = 
////  		  	outputEvaluator.determineBestVoiceAssignment(currentBasicTabSymbolPropertiesChord, currentNetworkOutputs,
////  		  	currentPossibleVoiceAssignments, isTrainingOrTestMode); TODO <-- veranderd voor SysMus
//		  	  outputEvaluator.determineBestVoiceAssignment(currentNetworkOutputs, currentPossibleVoiceAssignments); 
//  		  allBestVoiceAssignments.add(predictedBestVoiceAssignment);
//  		   
//  		  // d. Convert predictedBestVoiceAssignment into a List of voices, and add it to allPredictedVoices
//      	List<List<Double>> predictedChordVoiceLabels = dataConverter.getChordVoiceLabels(predictedBestVoiceAssignment); 
//      	List<List<Integer>> predictedChordVoices = dataConverter.getVoicesInChord(predictedChordVoiceLabels); 
//  		  allPredictedVoices.addAll(predictedChordVoices);
//      }
//    }   
//    return allPredictedVoices; 
		return null;
	}
	
	
	/**
	 * Determines whether the NN's performance criteria are met. This is the case when the trainingError is 
	 * smaller than its given threshold AND the classificationError is smaller than its given threshold. 
	 * 
	 * @param trainingError The training error
	 * @param classificationError The classification error 
	 * @param trainingErrorThreshold The value the trainingError should not exceed
	 * @param classificationErrorThreshold The value the classificationError should not exceed
	 * @return True when the performance criteria are met; false if not.
	 */
	private static boolean evaluatePerformanceNOTINUSE(double[] trainingError, double classificationError,
		double trainingErrorThreshold, double classificationErrorThreshold) {
		boolean performanceThresholdPassed = false;
		// Criteria for good performance:
		// a) if the error doesn’t change anymore
//		double trainingErrorThreshold = 0.01;
//		double classificationErrorThreshold = 0.1;
		double currentTrainingError = Math.abs(trainingError[0] - trainingError[1]);
		double currentClassificationError = classificationError;

		// Performance is good enough if trainingError < trainingErrorThreshold and
		// classificationError < classificationThreshold
		if (currentTrainingError < trainingErrorThreshold && currentClassificationError < classificationErrorThreshold) {
			System.out.println("Performance good enough.");
			performanceThresholdPassed = true;
		}
		return performanceThresholdPassed;
	}
	
	
	/**
	 * Gets the network output for noteFeatures or chordFeatures with the given settings.
	 * 
	 * @param learningApproach
	 * @param allowCoD
	 * @param deviationThreshold
	 * @return
	 */
	// FIXME commented out 12.11.2016
	private List<List<Double>> getNetworkOutputNOTINUSE(int learningApproach, double allowCoD,
		double deviationThreshold) {
//		List<List<Double>> allPredictedLabels = new ArrayList<List<Double>>();
//		
//    if (learningApproach == NOTE_CLASS) { 
//      int numberOfOnsets = noteFeatures.size();
//      for (int i = 0; i < numberOfOnsets; i++) {
//        // Evaluate the network for the current onset
//        double[] predictedLabel = evalNetwork(noteFeatures.get(i));
//        List<Double> predictedLabelAsList = ToolBox.convertArrayToList(predictedLabel);
//        allPredictedLabels.add(predictedLabelAsList);
//      }
//    }
//    if (learningApproach == CHORD_REGRESS) {
////  		int numberOfChords = chordFeatures.size(); 		
////  		// For each chord
////      for (int chordIndex = 0; chordIndex < numberOfChords; chordIndex++) { 			  
////		    callCount++;		
////      	// Get the predicted voice assignment
////        // a. For all possible features, evaluate the network and keep track of the highest value found. The index 
////      	// of the feature vector with the highest network output is the index in currentPossibleVoiceAssignment of
////      	// the predicted voice assignment
////      	List<List<Double>> currentChordFeatures = chordFeatures.get(chordIndex);
////      	List<List<Integer>> currentPossibleVoiceAssignments = possibleVoiceAssignmentsAllChords.get(chordIndex);
////  		  double highestOutput = Double.NEGATIVE_INFINITY; 
////      	int bestEval = -1;
////  		  for (int j = 0; j < currentChordFeatures.size(); j++) {
////  		  	List<Double> currentChordFeatureVector = currentChordFeatures.get(j);
////  		  	double[] currentOutput = evalNetwork(currentChordFeatureVector);
////  		  	if (currentOutput[0] > highestOutput) {
////    	 			highestOutput = currentOutput[0];
////    	 			bestEval = j;
////    	 		}
////  		  }
////  		  // b. Determine the predicted voice assignment, 
////  		  List<Integer> predictedVoiceAssignment = currentPossibleVoiceAssignments.get(bestEval);
////  		  // c. convert predictedVoiceAssignment into a List of voices, and add it to allPredictedVoices
////  		  // Convert predictedVoiceAssignment into a List of voices labels
////      	List<List<Double>> predictedChordVoiceLabels = dataConverter.getChordVoiceLabels(predictedVoiceAssignment); 
////      	// Convert predictedChordVoiceLabels into a List of voices
////      	List<List<Integer>> predictedChordVoices = dataConverter.getVoicesInChord(predictedChordVoiceLabels); 
////  		  allPredictedVoices.addAll(predictedChordVoices);
////      }
//    }
//    
//    return allPredictedLabels; 
		return null;
	}
	
	
	/**
	 * Returns a String containing, for the given learning approach, the label and network output details.
	 * 
	 * @param learningApproach
	 * @param dataConverter
	 * @return
	 */
	private String getLabelAndNetworkOutputDetailsTrainingNOTINUSE(Map<String, Double> argTrainingInfoMap, 
		int learningApproach/*, DataConverter dataConverter*/) {
		String stringToReturn = "";
//	  // Make a DecimalFormat for formatting long double values to four decimal places
//	  DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.UK);
//		otherSymbols.setDecimalSeparator('.'); 
//	 	DecimalFormat decForm = new DecimalFormat("#.####", otherSymbols);
//	 	
//	 	// Write out labels and network output details
////	 	String featuresString = null;
// 		String labelString = null;
// 		String outputString = null;
//	 	if (learningApproach == NOTE_TO_NOTE) {
//	 		for (int i = 0; i < onsetFeatures.size(); i++) {
//	 			List<Double> currentFeatures = onsetFeatures.get(i);
////	 			// Format features
////	 			featuresString = "";
////	   	  for (int j = 0; j < currentFeatures.size(); j++) {
////	   	    String df = decForm.format(currentFeatures.get(j));
////	   	    featuresString += df + "\t";
////	   	  }
//	   	  
//	 			stringToReturn += "onsetIndex = " + i + "\r\n";
//	   	   
//	 			// Format labels
//	 			stringToReturn += "label" + "\r\n";
//	 			labelString = "";
//	   	  List<Double> currentLabel = groundTruthVoiceLabels.get(i);
//	      List<Integer> actualVoices = dataConverter.convertIntoListOfVoices(currentLabel);
//	   	  for (int j = 0; j < currentLabel.size(); j++) {
//	   	    if (j == 0) {
//	   	    	labelString += "[" + currentLabel.get(j) + "\t";
//	   	    }
//	   	    else if (j == currentLabel.size() - 1) {
//	      		labelString += currentLabel.get(j) + "]" + " --> voice(s) " + actualVoices + "\r\n";  	 
//	   	    }
//	   	    else {
//	   	    	labelString += currentLabel.get(j) + "\t";
//	   	    }
//	      }
//	   	  stringToReturn += labelString;
//	      
//	      // Format output
//		   	stringToReturn += "network output" + "\r\n";
//	   	  outputString = "";
//		   	double[] currentOutput = evalNetwork(currentFeatures);
//		   	List<Integer> predictedVoices =	outputEvaluator.interpretPredictedLabel(currentOutput,
//		   		argTrainingInfoMap.get(ALLOW_COD), argTrainingInfoMap.get(DEVIATION_THRESHOLD));
//		   	for (int j = 0; j < currentOutput.length; j++) {
//		   		String df = decForm.format(currentOutput[j]);
//		      if (j == 0) {
//		      	outputString += "[" + df + "\t"; 
//		      }
//		      else if (j == currentOutput.length - 1) {
//		       	outputString += df + "]" + " --> voice(s) " + predictedVoices + "\r\n";
//		      }
//		      else {
//		       	outputString += df + "\t";
//		      }
//		   	}
//		   	stringToReturn += outputString + "\r\n";
//	 		}
//	  }
//	 	else if (learningApproach == CHORD_TO_CHORD) {
//	  	for (int i = 0; i < chordFeatures.size(); i++) {
//	  		List<List<Double>> currentChordFeatures = chordFeatures.get(i);
////	  		// Format features
////	  		featuresString = "  ";		
////	  		for (int j = 0; j < currentChordFeatures.size(); j++) {
////	   	    List<Double> currentOnsetFeatures = currentChordFeatures.get(j);
////	   	    for (int k = 0; k < currentOnsetFeatures.size(); k++) {
////	   	    	double currentOnsetFeature = currentOnsetFeatures.get(k);
////	   	      String df = decForm.format(currentOnsetFeature);
////	   	      featuresString += df + "\t";
////	   	    }
////	   	    featuresString += "\n";
////	   	  }
//	   	
//	  		stringToReturn += "chordIndex = " + i + "\r\n";
//	  		
//	  		// Format labels
//	  		stringToReturn += "labels" + "\r\n";
//	  		labelString = "";
//	  		List<List<Double>> currentChordLabels = groundTruthChordVoiceLabels.get(i); 
//        for (int j = 0; j < currentChordLabels.size(); j++) {
//        	List<Double> currentOnsetLabel = currentChordLabels.get(j);
//        	List<Integer> actualVoices = dataConverter.convertIntoListOfVoices(currentOnsetLabel);
//        	for (int k = 0; k < currentOnsetLabel.size(); k++) {
//		   	    if (k == 0) {
//		   	    	labelString += "[" + currentOnsetLabel.get(k) + "\t";
//		   	    }
//		   	    else if (k == currentOnsetLabel.size() - 1) {
//		   	    	labelString += currentOnsetLabel.get(k) + "]" + " --> voice(s) " + actualVoices + "\r\n"; 
//		   	    }
//		   	    else {
//		   	    	labelString += currentOnsetLabel.get(k) + "\t"; 
//		   	    }
//		      }
//        }
//        stringToReturn += labelString;
//        
//	  	  // Format output
//        stringToReturn += "highest network output" + "\r\n";
//        outputString = "";
//	  		double highestOutput = 0.0; // TODO OK? Is activation value always between 0.0 and 1.0?
//	  		List<List<Integer>> possibleVoiceAssignmentsCurrentChord = possibleVoiceAssignmentsAllChords.get(i);
//	  		int bestEval = -1;
//	  		for (int j = 0; j < currentChordFeatures.size(); j++) {
//	  			List<Double> currentChordFeatureVector = currentChordFeatures.get(j);	
//	  			double[] currentOutput = evalNetwork(currentChordFeatureVector);
//	  			if (currentOutput[0] > highestOutput) {
//    	 			highestOutput = currentOutput[0];
//    	 			bestEval = j;
//    	 		}	
//	  		}
//	  		List<Integer> bestEvalVoiceAssignment = possibleVoiceAssignmentsCurrentChord.get(bestEval);
//	  		List<List<Double>> bestEvalLabels = dataConverter.getChordVoiceLabels(bestEvalVoiceAssignment);
//	  		List<List<Integer>> predictedChordVoices = dataConverter.getVoicesInChord(bestEvalLabels);
////	  		for (int j = 0; j < bestEvalLabels.size(); j++) {
////        	List<Double> currentOnsetLabel = bestEvalLabels.get(j);
////        	List<Integer> currentPredictedVoices = dataConverter.convertIntoListOfVoices(currentOnsetLabel);
////        	for (int k = 0; k < currentOnsetLabel.size(); k++) {
////		   	    if (k == 0) {
////		   	    	outputString += "[" + currentOnsetLabel.get(k) + "\t"; 
////		   	    }
////		   	    else if (k == currentOnsetLabel.size() - 1) {
////		   	    	outputString += currentOnsetLabel.get(k) + "]" + " --> voice(s) " + currentPredictedVoices + "\r\n"; 
////		   	    }
////		   	    else {
////		   	      	outputString += currentOnsetLabel.get(k) + "\t"; 
////		   	      }
////		      }
////        }
//	  		outputString += "[" + decForm.format(highestOutput) + "] for voice assignment no. " + bestEval + ": " +
//		  		bestEvalVoiceAssignment + " --> voice(s) " + predictedChordVoices + "\r\n";
//	  		stringToReturn += outputString + "\r\n";
//	  	}
//	  }
	 	return stringToReturn;
	}
	
	/**
	 * Tests the network using the best weights obtained from the training.
	 * 
	 * @param bestWeightsFile The file containing the best weights
	 */
	// FIXME commented out 12.11.2016
//	private void testNetworkNOTINUSE(int learningApproach, int featureSet, String prefix, String pieceName,
//		/*DataConverter dataConverter, */ErrorCalculator argErrCalc, OutputEvaluator argOutputEvaluator, double allowCoD,
//		double deviationThreshold, int numberOfTestExamples) {
//		
////		String learningApproachAsString = null;
////		List<Double> bestWeights = null;
////		if (learningApproach == NOTE_TO_NOTE) {
////			learningApproachAsString = "NOTE_TO_NOTE";
//////			bestWeights = getWeights(new File(prefix + bestWeightsNoteToNoteSuffix));
////			bestWeights = 
////				AuxiliaryTool.getStoredObject(new ArrayList<Double>(), new File(prefix + bestWeightsNoteToNoteSuffix));
////		}
////		else if (learningApproach == CHORD_TO_CHORD) {
////			learningApproachAsString = "CHORD_TO_CHORD";
//////			bestWeights = getWeights(new File(prefix + bestWeightsChordToChordSuffix));
////			bestWeights = 
////				AuxiliaryTool.getStoredObject(new ArrayList<Double>(), new File(prefix + bestWeightsChordToChordSuffix));
////		}
////		
////		outputEvaluator = argOutputEvaluator;
////		errorCalculator = argErrCalc;
////		
////		testStartTime = AuxiliaryTool.getTimeStamp();	  
//////		initWeightsFromFile(bestWeightsFile);
////		initWeightsFromList(bestWeights);
////		List<List<Integer>> allPredictedVoices = determinePredictedVoices(learningApproach, allowCoD, deviationThreshold, 
////		  false, dataConverter, outputEvaluator);
////		List<List<Integer>> assignmentErrors = errorCalculator.calculateAssignmentErrors(allPredictedVoices, groundTruthVoiceLabels);
////		//		List<List<Integer>> assignmentErrors = calculateAssignmentErrors(learningApproach, allowCoD, deviationThreshold, 
//////		 	dataConverter, outputEvaluator, errorCalculator); 
////	  testEndTime = AuxiliaryTool.getTimeStamp();
////
////	  String[] info = new String[5];
////	  info[0] = testStartTime;
////	  info[1] = testEndTime;
////	  info[2] = pieceName;
////	  info[3] = prefix;
////	  info[4] = ""; // is empty in test case, contains conflictsRecord in application case
////
////	  String settings = getTestAndApplicationResults(learningApproach, assignmentErrors, bestWeights, info,
////	  	numberOfTestExamples,	null, null, errorCalculator, dataConverter, outputEvaluator);
//////    String fileName = prefix + "Test results " + learningApproachAsString + " " + pieceName + ".txt";
////	  String fileName = prefix + "Test process record " + learningApproachAsString + " " + pieceName + ".txt";
//////	  storeTestAndApplicationResults(settings, fileName);
////	  AuxiliaryTool.storeTextFile(settings, new File(fileName));
//	}
	
}
