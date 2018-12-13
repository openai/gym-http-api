use std::iter;

use serde_json::Value;
use serde_json::value::from_value;

use error::GymResult;

use rand::{thread_rng, Rng};

fn tuples(sizes: &[u64]) -> Vec<Vec<u64>> {
    match sizes.len() {
        0 => vec![],
        1 => (0..sizes[0]).map(|x| vec![x]).collect(),
        _ => {
            let (&head, tail) = sizes.split_first().unwrap();
            (0..head).flat_map(|x| iter::repeat(x).zip(tuples(tail))
                                             .map(|(h, mut t)| {
                                                t.insert(0, h);
                                                t
                                             })
                     ).collect()
        }
    }
}

#[derive(Debug, Clone)]
pub enum Space {
    DISCRETE{n: u64},
    BOX{shape: Vec<u64>, high: Vec<f64>, low: Vec<f64>},
    TUPLE{spaces: Vec<Box<Space>>}
}

impl Space {
    pub(crate) fn from_json(info: &Value) -> GymResult<Space> {
        match info["name"].as_str().unwrap() {
            "Discrete" => {
                let n = info["n"].as_u64().unwrap();
                Ok(Space::DISCRETE{n: n})
            },
            "Box" => {
                let shape = from_value(info["shape"].clone())?;
                let high = from_value(info["high"].clone())?;
                let low = from_value(info["low"].clone())?;

                Ok(Space::BOX{shape: shape, high: high, low: low})
            },
            "Tuple" => panic!("Parsing for Tuple spaces is not yet implemented"),
            e @ _ => panic!("Unrecognized space name: {}", e)
        }
    }
    pub fn sample(&self) -> Vec<f64> {
        let mut rng = thread_rng();
        match *self {
            Space::DISCRETE{n} => {
                vec![(rng.gen::<u64>()%n) as f64]
            },
            Space::BOX{ref shape, ref high, ref low} => {
                let mut ret = Vec::with_capacity(shape.iter().map(|x| *x as usize).product());
                let mut index = 0;

                for _ in tuples(shape) {
                    ret.push(rng.gen_range(low[index], high[index]));
                    index += 1
                }
                ret
            },
            Space::TUPLE{ref spaces} => {
                let mut ret = Vec::new();
                for space in spaces {
                    ret.extend(space.sample());
                }
                ret
            }
        }
    }
}