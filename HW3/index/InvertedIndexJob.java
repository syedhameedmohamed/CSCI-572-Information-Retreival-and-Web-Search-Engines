import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;

public class InvertedIndexJob {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {
        private Text word = new Text();
        private Text documentNumber = new Text();
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] fileContents = value.toString().split("\t", 2);
            String data = fileContents[1].toLowerCase();
            data = data.replaceAll("\\s+", " ");
            data = data.replaceAll("[^a-z\\s]", " ");

            StringTokenizer stringTokenizer = new StringTokenizer(data);
            documentNumber.set(fileContents[0]);

            while (stringTokenizer.hasMoreTokens()) {
                word.set(stringTokenizer.nextToken());
                context.write(word, documentNumber);
            }
        }
    }

    public static class IndexReducer extends Reducer<Text, Text, Text, Text> {
        private Text result = new Text();
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            HashMap<String, Integer> countTracker = new HashMap<>();
            for (Text value : values) {
                String documentNumber = value.toString();
                countTracker.put(documentNumber, countTracker.getOrDefault(documentNumber, 0) + 1);
            }

            StringBuilder finalString = new StringBuilder();
            for (String s : countTracker.keySet())
                finalString.append(s).append(":").append(countTracker.get(s)).append("\t");

            result.set(finalString.substring(0, finalString.length() - 1));
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Inverted Index");
        job.setJarByClass(InvertedIndexJob.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setReducerClass(IndexReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
