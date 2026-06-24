/** Simulated GNSS site */
export interface GnssSite {
  lon: number
  lat: number
  ve: number   // mm/yr
  vn: number
  se: number
  sn: number
  name: string
  isOutlier?: boolean
}

/** Simulated triangle result */
export interface StrainTriangle {
  lon0: number; lat0: number
  lon1: number; lat1: number
  lon2: number; lat2: number
  clon: number; clat: number
  dilatation: number   // nstrain/yr
  maxShear: number
  e1: number; e2: number   // principal strains (nstrain/yr)
  azimuth: number          // degrees: angle of e2 (extension) from East, CCW
  isGood: boolean
}

export interface DataBounds {
  lonMin: number; lonMax: number
  latMin: number; latMax: number
}

export interface ComputeParams {
  // data
  velFormat: 'auto' | 'gmt' | 'globk'
  outputDir: string
  // density
  minSpacingKm: number | null
  maxEdgeKm: number | null
  // triangulation
  minAngleDeg: number
  maxEdgePctl: number
  maxEdgeFactor: number
  // outlier
  kNeighbors: number
  madFactor: number
  iqrFactor: number
  maxOutlierIter: number
  // smoothing
  smoothWeight: number
  smoothIter: number
  // uncertainty
  mcIterations: number
}

export interface ComputeResult {
  rawSites: GnssSite[]
  cleanSites: GnssSite[]
  outlierSites: GnssSite[]
  triangles: StrainTriangle[]
  bounds: DataBounds
  stats: {
    nInput: number
    nClean: number
    nOutlier: number
    nTriangles: number
    nGoodTri: number
    dilatMin: number
    dilatMax: number
    shearMax: number
  }
  log: string[]
}

export const DEFAULT_PARAMS: ComputeParams = {
  velFormat: 'auto',
  outputDir: 'output',
  minSpacingKm: null,
  maxEdgeKm: null,
  minAngleDeg: 10,
  maxEdgePctl: 95,
  maxEdgeFactor: 1.5,
  kNeighbors: 8,
  madFactor: 3.5,
  iqrFactor: 1.5,
  maxOutlierIter: 5,
  smoothWeight: 0.3,
  smoothIter: 2,
  mcIterations: 200,
}
