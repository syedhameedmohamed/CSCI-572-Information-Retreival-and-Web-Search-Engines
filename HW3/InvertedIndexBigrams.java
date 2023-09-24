import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.StringTokenizer;

public class InvertedIndexBigrams {

    public static class BigramsInvertedIndexMapper extends Mapper<Object, Text, Text, Text> {

        private final Text bigram = new Text();
        private final Text docId = new Text();

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String firstWord = null;
            String secondWord = null;
            String[] fileContents = value.toString().split("\\t", 2);
            docId.set(fileContents[0]);

            String words = fileContents[1].replaceAll("[^a-zA-Z]+", " ").toLowerCase(Locale.ROOT);
            StringTokenizer tokenizer = new StringTokenizer(words, " ");

            while (tokenizer.hasMoreTokens()) {
                String wordString = tokenizer.nextToken();
                if (!wordString.trim().isEmpty()) {
                    if (firstWord == null) {
                        firstWord = wordString;
                        continue;
                    } else if (secondWord == null) {
                        secondWord = wordString;
                    } else {
                        firstWord = secondWord;
                        secondWord = wordString;
                    }
                    bigram.set(String.format("%s %s", firstWord, secondWord));

                    context.write(bigram, docId);
                }
            }
        }
    }

    public static class BigramsInvertedIndexReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        public void reduce(Text bigram, Iterable<Text> docIds, Context context) throws IOException, InterruptedException {
            Map<String, Integer> bigramCountTrackerMap = new HashMap<>();
            for (Text docId : docIds) {
                String docIdString = docId.toString();
                bigramCountTrackerMap.put(docIdString, bigramCountTrackerMap.getOrDefault(docIdString, 0) + 1);
            }

            StringBuilder finalBigramStringFrequency = new StringBuilder();
            for (Map.Entry<String, Integer> entry : bigramCountTrackerMap.entrySet()) {
                if (finalBigramStringFrequency.length() > 0) {
                    finalBigramStringFrequency.append("\t");
                }
                String docId = entry.getKey();
                Integer bigramFrequency = entry.getValue();
                String docIdBigramFrequency = String.format("%s:%d", docId, bigramFrequency);
                finalBigramStringFrequency.append(docIdBigramFrequency);
            }

            context.write(bigram, new Text(finalBigramStringFrequency.toString()));
        }
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        if (args.length != 2) {
            System.err.println("Usage: Bigrams Inverted Index <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Inverted Index Bigrams");
        job.setJarByClass(InvertedIndexBigrams.class);
        job.setMapperClass(BigramsInvertedIndexMapper.class);
        job.setReducerClass(BigramsInvertedIndexReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        Path inputFilePath = new Path(args[0]);
        Path outputFilePath = new Path(args[1]);
        FileSystem fileSystem = outputFilePath.getFileSystem(conf);
        if (fileSystem.exists(outputFilePath)) {
            fileSystem.delete(outputFilePath, true);
        }
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
