import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_dir', 
        help='Directory to input data in parquet format',
        type=str,
    )
    parser.add_argument(
        'output_dir',
        help='Directory to output data in tsv format',
        type=str,
    )
    args = parser.parse_args()

    spark = SparkSession.builder \
            .master('local[*]') \
            .config('spark.local.dir', '/local/shared/spark') \
            .config('spark.driver.memory', '32g') \
            .config('spark.sql.debug.maxToStringFields', 1000) \
            .appName('app') \
            .getOrCreate()

    print(f'Read from {args.input_dir}')
    df = spark.read.parquet(args.input_dir).orderBy('cyril_query_group_id')
    for name, dtype in df.dtypes:
        if dtype == 'string':
            assert df.filter(F.col(name).contains('\t')).count() == 0, \
                    'Replace tabs in column %s before writing into tsv' % name

    df.write.csv(args.output_dir, header=True, sep='\t', mode='overwrite')
    print(f'Write into {args.output_dir}')


if __name__ == '__main__':
    main()
