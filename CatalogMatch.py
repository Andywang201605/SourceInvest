
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import astropy.units as u

viziertab = {
    'GLEAM':{
        'catalog': ['VIII/100/gleamegc', 'VIII/102/gleamgal'],
        'freq': 0.2,
        'fluxcol': ['Fpwide', 'e_Fpwide'],
        'matchradius': 15.,
        'tomJy':1e3
    },
    'SUMSS':{
        'catalog': ['VIII/81B/sumss212', 'VIII/82/mgpscat'],
        'freq': 0.843,
        'fluxcol': ['Sp', 'e_Sp'],
        'matchradius': 15.,
        'tomJy':1
    },
    'NVSS':{
        'catalog': ['VIII/65/nvss'],
        'freq': 1.4,
        'fluxcol': ['S1.4', 'e_S1.4'],
        'matchradius': 15.,
        'tomJy':1
    },
    'FIRST':{
        'catalog': ['VIII/92/first14'],
        'freq': 1.4,
        'fluxcol': ['Fpeak', 'Rms'],
        'matchradius': 15.,
        'tomJy': 1
    },
    'VLASS':{
        'catalog': ['J/ApJS/255/30/comp'],
        'freq': 3.0,
        'fluxcol': ['Fpeak', 'e_Fpeak'],
        'matchradius': 15.,
        'tomJy': 1
    },
    'AT20G':{
        'catalog': ['J/MNRAS/402/2403/at20gcat'],
        'freq': [5.0, 8.0, 20.0],
        'fluxcol': [['S5', 'e_S5'], ['S8', 'e_S8'], ['S20', 'e_S20']],
        'matchradius': 15.,
        'tomJy': 1
    }
}

class CATMATCH():
    def __init__(
        self, catname, catalog=None, freq=None, fluxcol=None, 
        matchradius=None, tomJy=1.
    ):
        self.catname = catname
        ### catalogues
        self.catalog = catalog if isinstance(catalog, list) else [catalog]
        ### freq & flux col
        if isinstance(freq, list): self.freq = freq; self.fluxcol = fluxcol
        else: self.freq = [freq]; self.fluxcol = [fluxcol]

        ### match radius
        self.radius = matchradius
        self.tomJy = tomJy

        ### connect to vizier
        self.v = self._connectVizier_()
        
    def _connectVizier_(self):
        return Vizier(columns=["*", "+_r"], catalog=self.catalog)
    
    def _querysrc_(self, ra, dec):
        srccoord = SkyCoord(ra, dec, unit=u.degree)
        return self.v.query_region(srccoord, radius=f'{self.radius}s')
    
    def _parsetablist_(self, vtablist):
        '''Parse result from `Vizier.query_region`
        '''
        if len(vtablist) == 0: return self._parsetab_(None)

        return ''.join(self._parsetab_(vtab) for vtab in vtablist)
            
        
    def _parsetab_(self, vtab=None):
        vtabtext = ''
        # if no table found
        if vtab is None: 
            linehead = f'{self.catname},{self.radius},--,--'
            for freq in self.freq:
                vtabtext += f'{linehead},{freq},\n'
            return vtabtext
        
        # work out match separation
        sepunit = vtab['_r'].unit
        matchsep = ((vtab['_r'][0] * sepunit).to('arcsec')).value
        linehead = f'{self.catname},{matchsep}'
        for freq, cols in zip(self.freq, self.fluxcol):
            vtabtext += f'{linehead},{vtab[cols[0]][0]},{vtab[cols[1]][0]},{freq},\n'
        return vtabtext
        
    def _makeheader_(self):
        return 'catalog,sep,flux,e_flux,freq,note\n'