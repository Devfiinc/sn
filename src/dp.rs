extern crate nalgebra as na;
use na::{DMatrix};


pub struct MeasurementDMatrix {
    pub _vdi: DMatrix<f64>,
    pub _vdo: DMatrix<f64>,
    pub _epsilon : f64,
    _bound_c : f64,
    _noise_scale : f64,
}


impl MeasurementDMatrix {
    pub fn new(epsilon : f64) -> MeasurementDMatrix  {
        MeasurementDMatrix {
            _vdi: DMatrix::<f64>::zeros(0, 0),
            _vdo: DMatrix::<f64>::zeros(0, 0),
            _epsilon: epsilon,
            _bound_c : 0.0,
            _noise_scale : 0.0,
        }
    }

    pub fn initialize(&mut self, epsilon: f64, bound_c : f64, noise_scale : f64) {
        self._epsilon = epsilon;
        self._bound_c = bound_c;
        self._noise_scale = noise_scale;
    }


    fn laplace(&self, x : f64, mu: f64, b :f64) -> f64 {
        let xp = (x-mu).abs()/b;
        let laplace = (-xp).exp() / (2.0 * b);
        return laplace;
    }

    
    fn gaussian(&self, x : f64, mu: f64, b :f64) -> f64 {
        let xp = (x-mu).abs()/b;
        let gaussian = (-xp).exp() / (2.0 * b);
        return gaussian;
    }


    pub fn invoke(&mut self, vdi: DMatrix<f64>) -> DMatrix<f64> {

        let mut vdo = self.clip_gravdient(vdi.clone());
        vdo = self.add_noise(vdo.clone());

        return vdo.clone();
    }


    pub fn clip_gravdient(&mut self, vdi: DMatrix<f64>) -> DMatrix<f64> {
        let mut vdo = DMatrix::<f64>::zeros(vdi.nrows(), vdi.ncols());

        let mut norm : f64 = 0.0;
        for i in 0..vdi.nrows() {
            for j in 0..vdi.ncols() {
                norm = norm + vdi[(i,j)]*vdi[(i,j)];
            }
        }
        norm = norm.sqrt();

        let unit : f64 = 1.0;
        let normval : f64 = unit.max(norm/self._bound_c);

        for i in 0..vdi.nrows() {
            for j in 0..vdi.ncols() {
                vdo[(i,j)] = vdi[(i,j)]/normval;
            }
        }

        self._vdo = vdo.clone();
        return vdo.clone();
    }

    /*
    pub fn add_noise_laplace_2(&mut self, vdi: DMatrix<f64>) -> DMatrix<f64> {
        self._vdi = vdi.clone();
        let s = self._vdi.amax();
        let laplace = probability::vdistribution::Laplace::new(0.0, s / self._epsilon);

        self._vdo = vdi.clone();

        for i in 0..self._vdi.nrows() {
            for j in 0..self._vdi.ncols() {
                self._vdo[(i,j)] = self.laplace(rand::ranvdom::<f64>(), 0.0, s / self._epsilon);
            }
        }

        return self._vdo.clone();
    }
    */



    pub fn add_noise(&mut self, vdi: DMatrix<f64>) -> DMatrix<f64> {
        self._vdi = vdi.clone();
        let s = self._vdi.amax();

        //let a = rand::ranvdom::<f64>();
        //let b = self.laplace(a, 0.0, s / self._epsilon);

        self._vdo = vdi.clone();

        for i in 0..self._vdi.nrows() {
            for j in 0..self._vdi.ncols() {
                self._vdo[(i,j)] += self.laplace(rand::random::<f64>(), 0.0, s / self._epsilon);
            }
        }

        return self._vdo.clone();
    }
}
