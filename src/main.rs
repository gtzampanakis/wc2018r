/*
The World Cup 2018 has the following format.

 A1 ----+
     RS1|----+
 B2 ----+    |
          QF1|----+
 C1 ----+    |    |
     RS2|----+    |
 D2 ----+         |
               SF1|----+
 E1 ----+         |    |
     RS5|----+    |    |
 F2 ----+    |    |    |
          QF3|----+    |
 G1 ----+    |         |
     RS6|----+         |
 H2 ----+              |
                       |
                      F|------
                       |
 B1 ----+              |
     RS3|----+         |
 A2 ----+    |         |
          QF2|----+    |
 D1 ----+    |    |    |
     RS4|----+    |    |
 C2 ----+         |    |
               SF2|----+
 F1 ----+         |
     RS7|----+    |
 E2 ----+    |    |
          QF4|----+
 H1 ----+    |
     RS8|----+
 G2 --- +

C1: winner of group C
F2: runner-up of group F
RS3: 3rd match in the Round of Sixteen
QF2: 2nd match in the QuarterFinals
SF1: 1st match in the SemiFinals
F: Final

Some convenient properties occur from this format.

In the analysis that follows the suffixes A and B are appended to the match
codes. For example, QF2A/QF2B means team A/B (top/bottom team as it appears in
the graph).

P(SF_1A, SF_2A)
   = \sum_{GA} P(SF_1A, SF_2A, GA_ABCD)
   = \sum_{GA} P(SF_1A, SF_2A | GA_ABCD) P(GA_ABCD)
   = \sum_{GA} P(SF_1A | GA_ABCD) P(SF_2A | GA_ABCD)

P(SF_1B, SF_2B) = \sum_{GA P(SF_1B | GA_EFGH) P(SF_2B | GA_EFGH) P(GA_EFGH)

P(SF_1A, SF_1B, SF_2A, SF_2B) = P(SF_1A, SF_2A) P(SF_1B, SF_2B)

P(F) = \sum_{SF} P(F | SF1A, SF1B, SF2A, SF2B) P(SF1A, SF2A) P(SF1B, SF2B)

*/

use std::collections::HashMap;
use std::cmp::Eq;
use std::hash::Hash;

/* A distribution is a Vec of pairs, each pair being (event, prob). */

type Dist<T> = HashMap<T, f64>;
type Merits<'a> = HashMap<&'a str, f64>;
type Teams<'a> = Vec<&'a str>;


fn prod(ns: &Vec<f64>) -> f64 {
    let mut result = 1.;
    for n in ns {
        result *= n;
    }
    result
}

fn sum(ns: &Vec<f64>) -> f64 {
    let mut result = 0.;
    for n in ns {
        result += n;
    }
    result
}


/* Combines independent distributions into one. The resulting distribution has
 * references to the originals' keys. */
fn comb_dists<T: Eq + Hash + Clone>(ds: &[ Dist<T> ]) -> Dist<Vec<T>> 
{
    let mut result = Dist::new();
    
    let dist_vec_2d: Vec<Vec<(&T, &f64)>> = ds.iter().map(
        |d| d.iter().collect()
    ).collect();

    let mut productor = CartesianProductor::new(
        ds.iter().map(|d| d.len()).collect()
    );

    while productor.next() {
        let mut new_key: Vec<T> = Vec::new();
        let mut new_prob: f64 = 1.;

        for (k, &i) in productor.v.iter().enumerate() {
            new_key.push( (*dist_vec_2d[k][i].0).clone() );
            new_prob *= dist_vec_2d[k][i].1;
        }

        result.insert(new_key, new_prob);
    }

    result
}


/* Joint probability of group assignments. */
fn p_ga<'a>(gs: &[&Teams<'a>], m: &Merits<'a>) 
                                    -> Dist<Vec<(&'a str, &'a str)>> {
    let mut dists = Vec::new();

    for g in gs {
        dists.push(group_probs(g, m));
    }

    let result = comb_dists(&dists);

    result
}


/*
j does not make a difference here so it is not required in the arguments.
You just have to pass the correct groups, [a,b,c,d] for j=1 or [e,f,g,h] for
j=2.
*/
fn p_sfij_given_ga<'a>(i: usize, ga: &[(&'a str, &'a str)], m: &Merits) 
                                                            -> Dist<&'a str>
{
    if i == 1 {
        bracket_conv(
            &vec![ga[0].0, ga[1].1, ga[2].0, ga[3].1],
            m
        )
    }
    else if i == 2 {
        bracket_conv(
            &vec![ga[1].0, ga[0].1, ga[3].0, ga[2].1],
            m
        )
    }
    else {
        panic!("p_sfij_given_ga: Unexpected value for i: {}", i);
    }
}


fn p_sf1j_sf2j<'a>(gs: &[&Teams<'a>], m: &Merits<'a>) 
                                            -> Dist<(&'a str, &'a str)> {
    let mut result = Dist::new();

    for (ga, pga) in p_ga(gs, m).iter() {
        let d1 = p_sfij_given_ga(1, ga, m);
        let d2 = p_sfij_given_ga(2, ga, m);
        let d12 = comb_dists(&[d1, d2]);

        for (sfpair, psfpair) in d12 {
            update_dist(&mut result, (sfpair[0], sfpair[1]), psfpair * pga);
        }
    }

    result
}


/* Joint probability of SF1A, SF2A, SF1B, SF2B.  */
fn p_sf1a_sf2a_sf1b_sf2b<'a>(gs: &[&Teams<'a>], m: &Merits<'a>) 
                        -> Dist<Vec<(&'a str, &'a str)>> {
    comb_dists(
        &[
            p_sf1j_sf2j(&gs[..4], m),
            p_sf1j_sf2j(&gs[4..], m),
        ]
    )
}


/* Probability for the winner. */
fn p_f<'a>(gs: &[&Teams<'a>], m: &Merits<'a>) -> Dist<&'a str> {
    let sf_dist = p_sf1a_sf2a_sf1b_sf2b(gs, m);

    let mut result = Dist::new();

    for (sfpairs, psfpairs) in sf_dist {
        let d = bracket_conv(
            &[
                sfpairs[0].0,
                sfpairs[1].0,
                sfpairs[0].1,
                sfpairs[1].1,
            ],
            m
        );

        for (winner, pwinner) in d {
            update_dist(&mut result, winner, pwinner * psfpairs);
        }
    }

    result
}


fn update_dist<T: Eq + Hash>(d: &mut Dist<T>, k: T, v: f64) {
    d.entry(k)
        .and_modify(|old_v| *old_v += v)
        .or_insert(v);
}


struct Combinator {
    n: usize,
    k: usize,
    v: Vec<usize>,
    ret_first: bool,
}


impl Combinator {

    fn new(n: usize, k: usize) -> Combinator{
        Combinator {
            n: n,
            k: k,
            v: (0..k).collect(),
            ret_first: false,
        }
    }

    fn inc(&mut self, w: usize) {
        let max = self.n - (self.k - w);

        if self.v[w] == max {
            self.inc(w - 1);
            self.v[w] = self.v[w-1] + 1;
        }
        else {
            self.v[w] = self.v[w] + 1;
        }
    }

    fn next(&mut self) -> bool {
        if !self.ret_first {
            self.ret_first = true;
            return true;
        }

        if self.v[0] == self.n - self.k {
            return false;
        }

        let w = self.k - 1;
        self.inc(w);
        return true;
    }

}


struct Permuter {
    n: usize,
    i: usize,
    v: Vec<usize>,
    zs: usize,
}


impl Permuter {

    fn new(n: usize) -> Permuter {
        Permuter {
            n: n,
            i: 0,
            v: (0..n).collect(),
            zs: 2,
        }
    }

    fn next(&mut self) -> bool {
        let mut z = self.n;
        let mut w;

        self.v = Vec::new(); // TODO: make it so reallocation is not needed

        for a in 0..self.n {
            w = self.i % (a+1);

            if w == 0 {
                z -= 1;
            }
            self.v.insert(w, self.n-1-a);
        }

        if z == 0 {
            self.zs -= 1;
        }

        if self.zs == 0 {
            return false;
        }

        self.i += 1;

        return true;
    }

}


struct CartesianProductor {
    lens: Vec<usize>, // contains the lengths of the iterables
    v: Vec<usize>, // contains the results (updated for each next() call)
    done: bool,
    ret_first: bool,
}


impl CartesianProductor {
    fn new(lens: Vec<usize>) -> Self {
        let v = (0..lens.len()).map(|_| 0).collect();

        let result = Self {
            lens: lens,
            v: v,
            done: false,
            ret_first: false,
        };

        result
    }


    fn inc(&mut self, i: usize) {
        if self.v[i] == self.lens[i] - 1 {
            if i == 0 {
                self.done = true;
                return
            }
            self.v[i] = 0;
            self.inc(i-1);
        }
        else {
            self.v[i] += 1;
        }
    }


    fn next(&mut self) -> bool {
        if self.done {
            false
        }
        else if !self.ret_first {
            self.ret_first = true;
            true
        }
        else {
            let i = self.lens.len() - 1;
            self.inc(i);
            !self.done
        }
    }
}


fn rearrange<'a, T>(vin: &'a Vec<T>, idxs: &Vec<usize>, vout: &mut Vec<&'a T>) {
    for (idxi, idx) in idxs.iter().enumerate() {
        vout[idxi] = &vin[*idx];
    }
}



fn prob_match(t1: &str, t2: &str, m: &Merits) -> f64 {
    let m1 = m[t1];
    let m2 = m[t2];
    m1 / (m1 + m2)
}


fn group_probs<'a>(ts: &Teams<'a>, m: &Merits<'a>) 
                                -> Dist<(&'a str, &'a str)> {
    let mut sum_ms = 0.;
    let mut result = Dist::new();
    let mut c = Combinator::new(ts.len(), 2);

    for t in ts.iter() {
        sum_ms += m[t];
    }

    while c.next() {
        for si in 0..2 {
            let t1 = ts[c.v[si]];
            let t2 = ts[c.v[1-si]];
            let p1 = m[t1] / sum_ms;
            let p2 = m[t2] / (sum_ms - m[t1]);
            /* Keys are unique, no need to check if they already exist. */
            result.insert( (t1, t2), p1 * p2 );
        }
    }

    result
}


/* Convenience function that wraps the passed strings into a Dist of one
 * element. */
fn bracket_conv<'a>(ts: &[ &'a str ], m: &Merits) -> Dist<&'a str> {
    let mut v = Vec::new();
    for s in ts {
        let mut d = Dist::new();
        d.insert(*s, 1.);
        v.push(d);
    }
    return bracket(&v, m);
}


/* ts refers to a Vec of Dists. It represents the distribution of each team
 * participating in the bracket. */
fn bracket<'a>(ts: &[ Dist<&'a str> ], m: &Merits)
                                            -> Dist<&'a str> {
    let mut result;

    if ts.len() == 1 {
        result = ts[0].clone();
    }
    else if ts.len() == 2 {
        result = Dist::new();
        for (t1, p1) in &ts[0] {
            for (t2, p2) in &ts[1] {
                update_dist(&mut result, t1, p1 * p2 * prob_match(t1, t2, m));
                update_dist(&mut result, t2, p1 * p2 * prob_match(t2, t1, m));
            }
        }
    }
    else {
        let left = &ts[..(ts.len()/2)];
        let right = &ts[(ts.len()/2)..];

        let left_dist = bracket(left, m);
        let right_dist = bracket(right, m);

        result = bracket(
            &[left_dist, right_dist],
            m
        );
    }

    result
}


fn main() {
    let mut c = Combinator::new(5, 2);
    let vin = vec!["a", "b", "c", "d", "e"];
    let mut vout = vec![&"a", &"b"];

    while c.next() {
        println!("{:?}", c.v);
        rearrange(&vin, &c.v, &mut vout);
        println!("{:?}\n", vout);
    }

    c.ret_first = false;

    while c.next() {
        println!("{:?}", c.v);
        rearrange(&vin, &c.v, &mut vout);
        println!("{:?}\n", vout);
    }

}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::hash::Hash;
	use prod;
	use sum;
    use Merits;
    use bracket;
    use bracket_conv;
    use comb_dists;
    use group_probs;
    use p_f;
    use p_ga;
    use p_sf1a_sf2a_sf1b_sf2b;
    use p_sf1j_sf2j;
    use p_sfij_given_ga;
    use prob_match;
    use CartesianProductor;
    use Dist;

    const MODICUM: f64 = 1e-12;

    fn assert_dist_probs_equal_to_1<T: Eq + Hash>(d: &Dist<T>) {
        let s = d.values().fold(
            0.,
            |a, v| a + v
        );

        assert!((s-1.).abs() < MODICUM);
    }

    #[test]
    fn test_prod() {
		assert_eq!(prod(&vec![]), 1.);
		assert_eq!(prod(&vec![1., 2., 3.]), 6.);
		assert_eq!(prod(&vec![-1., 2., 3.]), -6.);
		assert_eq!(prod(&vec![-1., 2., -3.]), 6.);
    }

    #[test]
    fn test_sum() {
		assert_eq!(sum(&vec![]), 0.);
		assert_eq!(sum(&vec![1., 2., 3.]), 6.);
		assert_eq!(sum(&vec![-1., 2., 3.]), 4.);
		assert_eq!(sum(&vec![-1., 2., -3.]), -2.);
    }

    #[test]
    fn test_prob_match() {
        let mut m: Merits = HashMap::new();
        m.insert("1", 30.);
        m.insert("2", 10.);

        assert_eq!(prob_match("1", "1", &m), 0.50);
        assert_eq!(prob_match("1", "2", &m), 0.75);
        assert_eq!(prob_match("2", "1", &m), 0.25);
    }

    #[test]
    fn test_group_probs() {
        let mut m = Merits::new();
        m.insert("1", 30.);
        m.insert("2", 20.);
        m.insert("3", 10.);
        m.insert("4",  5.);

        let ts = vec!["1", "2", "3", "4"];

        let d = group_probs(&ts, &m);

        let mut s = 0.;

        for dm in d.iter() {
            s += dm.1;
        }

        assert!((s - 1.).abs() < MODICUM);
    }

    #[test]
    fn test_bracket() {
        let mut m = Merits::new();
        m.insert("1", 30.);
        m.insert("2", 20.);
        m.insert("3", 10.);
        m.insert("4",  4.);

        let d0 = bracket_conv(&["1"], &m);
        assert_eq!(d0["1"], 1.);

        let d1 = bracket_conv(&["1", "2"], &m);
        assert_eq!(d1["1"], 0.6);
        assert_eq!(d1["2"], 0.4);

        let d2 = bracket_conv(&["3", "4"], &m);
        assert_eq!(d2["3"], 0.7142857142857143);
        assert_eq!(d2["4"], 0.2857142857142857);

        let d11 = bracket(&[d1, d2], &m);
        assert_eq!(d11["1"], 0.47268907563025203);
        assert_eq!(d11["2"], 0.2857142857142857);
        assert_eq!(d11["3"], 0.20238095238095238);
        assert_eq!(d11["4"], 0.0392156862745098);
    }

    #[test]
    fn test_productor_0() {
        let lens = vec![1];
        let mut cp = CartesianProductor::new(lens);

        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0]);
        assert_eq!(cp.next(), false);
        assert_eq!(cp.next(), false);
    }

    #[test]
    fn test_productor_1() {
        let lens = vec![4];
        let mut cp = CartesianProductor::new(lens);

        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![3]);
        assert_eq!(cp.next(), false);
        assert_eq!(cp.next(), false);
    }

    #[test]
    fn test_productor_2() {
        let lens = vec![4, 2];
        let mut cp = CartesianProductor::new(lens);

        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![3, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![3, 1]);
        assert_eq!(cp.next(), false);
        assert_eq!(cp.next(), false);
    }

    #[test]
    fn test_productor_3() {
        let lens = vec![3, 4, 2];
        let mut cp = CartesianProductor::new(lens);

        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 0, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 0, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 1, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 1, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 2, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 2, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 3, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![0, 3, 1]);

        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 0, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 0, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 1, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 1, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 2, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 2, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 3, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![1, 3, 1]);

        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 0, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 0, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 1, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 1, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 2, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 2, 1]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 3, 0]);
        assert_eq!(cp.next(), true);
        assert_eq!(cp.v, vec![2, 3, 1]);

        assert_eq!(cp.next(), false);
        assert_eq!(cp.next(), false);
    }

    #[test]
    fn test_comb_dists_1() {
        let mut d1 = Dist::new();
        d1.insert("1", 1.);

        let ds = vec![d1];
        let combined = comb_dists(&ds);
        let prob = combined[&vec!["1"]];
        assert_eq!(prob, 1.);

        assert_dist_probs_equal_to_1(&combined);
    }

    #[test]
    fn test_comb_dists_2() {
        let mut d1 = Dist::new();
        d1.insert("1", 1.);
        let mut d2 = Dist::new();
        d2.insert("2", 1.);

        let ds = vec![d1, d2];
        let combined = comb_dists(&ds);
        let prob = combined[&vec!["1", "2"]];
        assert_eq!(prob, 1.);

        assert_dist_probs_equal_to_1(&combined);
    }

    #[test]
    fn test_comb_dists_3() {
        let mut d1 = Dist::new();
        d1.insert("1", 0.6);
        d1.insert("2", 0.4);

        let mut d2 = Dist::new();
        d2.insert("a", 0.7);
        d2.insert("b", 0.3);

        let ds = vec![d1, d2];
        let combined = comb_dists(&ds);

        let prob = combined[&vec!["1", "a"]];
        assert_eq!(prob, 0.42);

        let prob = combined[&vec!["1", "b"]];
        assert_eq!(prob, 0.18);

        let prob = combined[&vec!["2", "a"]];
        assert_eq!(prob, 0.27999999999999997);

        let prob = combined[&vec!["2", "b"]];
        assert_eq!(prob, 0.12);

        assert_dist_probs_equal_to_1(&combined);
    }

    #[test]
    fn test_p_ga() {
        let mut m = Merits::new();
        m.insert("a",  5.0);
        m.insert("b",  8.0);
        m.insert("c", 10.0);
        m.insert("d", 15.0);
        m.insert("e", 18.0);
        m.insert("f", 21.0);

        let d = p_ga(
            &vec![&vec!["a", "b", "c"], &vec!["d", "e", "f"]],
            &m
        );

        assert_eq!(d[&vec![("a", "b"), ("d", "e")]], 0.012386968908708041);
        assert_eq!(d[&vec![("a", "b"), ("d", "f")]], 0.014451463726826045);
        assert_eq!(d[&vec![("a", "b"), ("e", "d")]], 0.013419216317767043);
        assert_eq!(d[&vec![("a", "b"), ("e", "f")]], 0.01878690284487386);
        assert_eq!(d[&vec![("a", "b"), ("f", "d")]], 0.017079002586248962);
        assert_eq!(d[&vec![("a", "b"), ("f", "e")]], 0.020494803103498754);
        assert_eq!(d[&vec![("a", "c"), ("d", "e")]], 0.01548371113588505);
        assert_eq!(d[&vec![("a", "c"), ("d", "f")]], 0.018064329658532555);
        assert_eq!(d[&vec![("a", "c"), ("e", "d")]], 0.016774020397208805);
        assert_eq!(d[&vec![("a", "c"), ("e", "f")]], 0.023483628556092324);
        assert_eq!(d[&vec![("a", "c"), ("f", "d")]], 0.021348753232811202);
        assert_eq!(d[&vec![("a", "c"), ("f", "e")]], 0.02561850387937344);
        assert_eq!(d[&vec![("b", "a"), ("d", "e")]], 0.014864362690449648);
        assert_eq!(d[&vec![("b", "a"), ("d", "f")]], 0.01734175647219125);
        assert_eq!(d[&vec![("b", "a"), ("e", "d")]], 0.01610305958132045);
        assert_eq!(d[&vec![("b", "a"), ("e", "f")]], 0.02254428341384863);
        assert_eq!(d[&vec![("b", "a"), ("f", "d")]], 0.020494803103498754);
        assert_eq!(d[&vec![("b", "a"), ("f", "e")]], 0.0245937637241985);
        assert_eq!(d[&vec![("b", "c"), ("d", "e")]], 0.029728725380899296);
        assert_eq!(d[&vec![("b", "c"), ("d", "f")]], 0.0346835129443825);
        assert_eq!(d[&vec![("b", "c"), ("e", "d")]], 0.0322061191626409);
        assert_eq!(d[&vec![("b", "c"), ("e", "f")]], 0.04508856682769726);
        assert_eq!(d[&vec![("b", "c"), ("f", "d")]], 0.04098960620699751);
        assert_eq!(d[&vec![("b", "c"), ("f", "e")]], 0.049187527448397);
        assert_eq!(d[&vec![("c", "a"), ("d", "e")]], 0.021438984649686993);
        assert_eq!(d[&vec![("c", "a"), ("d", "f")]], 0.025012148757968155);
        assert_eq!(d[&vec![("c", "a"), ("e", "d")]], 0.023225566703827576);
        assert_eq!(d[&vec![("c", "a"), ("e", "f")]], 0.0325157933853586);
        assert_eq!(d[&vec![("c", "a"), ("f", "d")]], 0.02955981216850782);
        assert_eq!(d[&vec![("c", "a"), ("f", "e")]], 0.035471774602209384);
        assert_eq!(d[&vec![("c", "b"), ("d", "e")]], 0.03430237543949919);
        assert_eq!(d[&vec![("c", "b"), ("d", "f")]], 0.04001943801274905);
        assert_eq!(d[&vec![("c", "b"), ("e", "d")]], 0.03716090672612412);
        assert_eq!(d[&vec![("c", "b"), ("e", "f")]], 0.05202526941657376);
        assert_eq!(d[&vec![("c", "b"), ("f", "d")]], 0.04729569946961251);
        assert_eq!(d[&vec![("c", "b"), ("f", "e")]], 0.05675483936353501);

        assert_dist_probs_equal_to_1(&d);
    }

    #[test]
    fn test_p_sfij_given_ga() {
        let mut m = Merits::new();
        m.insert("a",  1.);
        m.insert("b",  2.);
        m.insert("c",  3.);
        m.insert("d",  4.);
        m.insert("e",  5.);
        m.insert("f",  6.);
        m.insert("g",  7.);
        m.insert("h",  8.);
        m.insert("i",  9.);
        m.insert("j", 10.);
        m.insert("k", 11.);
        m.insert("l", 12.);
        m.insert("m", 13.);
        m.insert("n", 14.);
        m.insert("o", 15.);
        m.insert("p", 16.);

        let d = p_sfij_given_ga(
            1,
            &vec![
                ("a", "b"),
                ("e", "f"),
                ("i", "j"),
                ("m", "n"),
            ],
            &m
        );

        assert_eq!(d["a"], 0.011387163561076604);
        assert_eq!(d["i"], 0.25155279503105593);
        assert_eq!(d["f"], 0.29068322981366457);
        assert_eq!(d["n"], 0.44637681159420284);

        assert_dist_probs_equal_to_1(&d);
    }

    #[test]
    fn test_p_sf1j_sf2j() {
        let mut m = Dist::new();
        m.insert("a",  1.);
        m.insert("b",  2.);
        m.insert("c",  3.);
        m.insert("d",  4.);
        m.insert("e",  5.);
        m.insert("f",  6.);
        m.insert("g",  7.);
        m.insert("h",  8.);
        m.insert("i",  9.);

        let d = p_sf1j_sf2j(
            &[
                &vec!["a", "b"],
                &vec!["c", "d"],
                &vec!["e", "f"],
                &vec!["g", "h"],
            ],
            &m
        );

        assert_dist_probs_equal_to_1(&d);

        assert!(
            (d[&("h", "g")] - 0.08104988102520536).abs() < MODICUM
        );
    }

    #[test]
    fn test_p_sf1a_sf2a_sf1b_sf2b() {
        let mut m = Merits::new();
        m.insert("a",  1.);
        m.insert("b",  2.);
        m.insert("c",  3.);
        m.insert("d",  4.);
        m.insert("e",  5.);
        m.insert("f",  6.);
        m.insert("g",  7.);
        m.insert("h",  8.);
        m.insert("i",  9.);
        m.insert("j", 10.);
        m.insert("k", 11.);
        m.insert("l", 12.);
        m.insert("m", 13.);
        m.insert("n", 14.);
        m.insert("o", 15.);
        m.insert("p", 16.);

        let d = p_sf1a_sf2a_sf1b_sf2b(
            &[
                &vec!["a", "b"],
                &vec!["c", "d"],
                &vec!["e", "f"],
                &vec!["g", "h"],
                &vec!["i", "j"],
                &vec!["k", "l"],
                &vec!["m", "n"],
                &vec!["o", "p"],
            ],
            &m
        );

        assert_dist_probs_equal_to_1(&d);

        assert!(
            (d[&vec![("h", "g"), ("p", "o")]] - 
                0.003958437863589254).abs() < MODICUM
        );
    }

    #[test]
    fn test_p_f() {
        let mut m = Merits::new();
        m.insert("a",  1.);
        m.insert("b",  2.);
        m.insert("c",  3.);
        m.insert("d",  4.);
        m.insert("e",  5.);
        m.insert("f",  6.);
        m.insert("g",  7.);
        m.insert("h",  8.);
        m.insert("i",  9.);
        m.insert("j", 10.);
        m.insert("k", 11.);
        m.insert("l", 12.);
        m.insert("m", 13.);
        m.insert("n", 14.);
        m.insert("o", 15.);
        m.insert("p", 16.);

        let d = p_f(
            &[
                &vec!["a", "b"],
                &vec!["c", "d"],
                &vec!["e", "f"],
                &vec!["g", "h"],
                &vec!["i", "j"],
                &vec!["k", "l"],
                &vec!["m", "n"],
                &vec!["o", "p"],
            ],
            &m
        );

        assert_dist_probs_equal_to_1(&d);

        assert!((d["p"] - 0.148055327561396).abs() < MODICUM);
    }
}
