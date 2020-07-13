/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package firtree.utilities;

import java.util.Random;
import java.util.logging.Logger;

/**
 * @author Van Dang
 */
public class MergeSorter {
    private static final Logger logger = Logger.getLogger(MergeSorter.class.getName());

    public static void main(final String[] args) {
        final float[][] f = new float[1000][];
        for (int r = 0; r < f.length; r++) {
            f[r] = new float[500];
            final Random rd = new Random();
            for (int i = 0; i < f[r].length; i++) {
                final float x = rd.nextInt(10);
                f[r][i] = x;
            }
        }
        final double start = System.nanoTime();
        for (final float[] element : f) {
            sort(element, false);
        }
        final double end = System.nanoTime();
        logger.info(() -> "# " + (end - start) / 1e9 + " ");
    }

    public static int[] sort(final float[] list, final boolean asc) {
        return sort(list, 0, list.length - 1, asc);
    }

    public static int[] sort(final float[] list, final int begin, final int end, final boolean asc) {
        final int len = end - begin + 1;
        final int[] idx = new int[len];
        final int[] tmp = new int[len];
        for (int i = begin; i <= end; i++) {
            idx[i - begin] = i;
        }

        //identify natural runs and merge them (first iteration)
        int i = 1;
        int j = 0;
        int k = 0;
        int start = 0;
        final int[] ph = new int[len / 2 + 3];
        ph[0] = 0;
        int p = 1;
        do {
            start = i - 1;
            while (i < idx.length
                    && ((asc && list[begin + i] >= list[begin + i - 1]) || (!asc && list[begin + i] <= list[begin + i - 1]))) {
                i++;
            }
            if (i == idx.length) {
                System.arraycopy(idx, start, tmp, k, i - start);
                k = i;
            } else {
                j = i + 1;
                while (j < idx.length
                        && ((asc && list[begin + j] >= list[begin + j - 1]) || (!asc && list[begin + j] <= list[begin + j - 1]))) {
                    j++;
                }
                merge(list, idx, start, i - 1, i, j - 1, tmp, k, asc);
                i = j + 1;
                k = j;
            }
            ph[p++] = k;
        } while (k < idx.length);
        System.arraycopy(tmp, 0, idx, 0, idx.length);

        //subsequent iterations
        while (p > 2) {
            if (p % 2 == 0) {
                ph[p++] = idx.length;
            }
            k = 0;
            int np = 1;
            for (int w = 0; w < p - 1; w += 2) {
                merge(list, idx, ph[w], ph[w + 1] - 1, ph[w + 1], ph[w + 2] - 1, tmp, k, asc);
                k = ph[w + 2];
                ph[np++] = k;
            }
            p = np;
            System.arraycopy(tmp, 0, idx, 0, idx.length);
        }
        return idx;
    }

    private static void merge(final float[] list, final int[] idx, final int s1, final int e1, final int s2, final int e2, final int[] tmp,
            final int l, final boolean asc) {
        int i = s1;
        int j = s2;
        int k = l;
        while (i <= e1 && j <= e2) {
            if (asc) {
                if (list[idx[i]] <= list[idx[j]]) {
                    tmp[k++] = idx[i++];
                } else {
                    tmp[k++] = idx[j++];
                }
            } else {
                if (list[idx[i]] >= list[idx[j]]) {
                    tmp[k++] = idx[i++];
                } else {
                    tmp[k++] = idx[j++];
                }
            }
        }
        while (i <= e1) {
            tmp[k++] = idx[i++];
        }
        while (j <= e2) {
            tmp[k++] = idx[j++];
        }
    }

    public static int[] sort(final double[] list, final boolean asc) {
        return sort(list, 0, list.length - 1, asc);
    }

    public static int[] sort(final double[] list, final int begin, final int end, final boolean asc) {
        final int len = end - begin + 1;
        final int[] idx = new int[len];
        final int[] tmp = new int[len];
        for (int i = begin; i <= end; i++) {
            idx[i - begin] = i;
        }

        //identify natural runs and merge them (first iteration)
        int i = 1;
        int j = 0;
        int k = 0;
        int start = 0;
        final int[] ph = new int[len / 2 + 3];
        ph[0] = 0;
        int p = 1;
        do {
            start = i - 1;
            while (i < idx.length
                    && ((asc && list[begin + i] >= list[begin + i - 1]) || (!asc && list[begin + i] <= list[begin + i - 1]))) {
                i++;
            }
            if (i == idx.length) {
                System.arraycopy(idx, start, tmp, k, i - start);
                k = i;
            } else {
                j = i + 1;
                while (j < idx.length
                        && ((asc && list[begin + j] >= list[begin + j - 1]) || (!asc && list[begin + j] <= list[begin + j - 1]))) {
                    j++;
                }
                merge(list, idx, start, i - 1, i, j - 1, tmp, k, asc);
                i = j + 1;
                k = j;
            }
            ph[p++] = k;
        } while (k < idx.length);
        System.arraycopy(tmp, 0, idx, 0, idx.length);

        //subsequent iterations
        while (p > 2) {
            if (p % 2 == 0) {
                ph[p++] = idx.length;
            }
            k = 0;
            int np = 1;
            for (int w = 0; w < p - 1; w += 2) {
                merge(list, idx, ph[w], ph[w + 1] - 1, ph[w + 1], ph[w + 2] - 1, tmp, k, asc);
                k = ph[w + 2];
                ph[np++] = k;
            }
            p = np;
            System.arraycopy(tmp, 0, idx, 0, idx.length);
        }
        return idx;
    }

    private static void merge(final double[] list, final int[] idx, final int s1, final int e1, final int s2, final int e2, final int[] tmp,
            final int l, final boolean asc) {
        int i = s1;
        int j = s2;
        int k = l;
        while (i <= e1 && j <= e2) {
            if (asc) {
                if (list[idx[i]] <= list[idx[j]]) {
                    tmp[k++] = idx[i++];
                } else {
                    tmp[k++] = idx[j++];
                }
            } else {
                if (list[idx[i]] >= list[idx[j]]) {
                    tmp[k++] = idx[i++];
                } else {
                    tmp[k++] = idx[j++];
                }
            }
        }
        while (i <= e1) {
            tmp[k++] = idx[i++];
        }
        while (j <= e2) {
            tmp[k++] = idx[j++];
        }
    }
}
