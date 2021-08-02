"""
function this = general_input(varargin)

this.name            = 'multi_feature_stimulus';
this.abbrev          = 'gen_inp';
this.description     = 'input distribution with many features';
this.d               = 4;
this.Tstim           = -1; % if set to -1 we set Tstim to Tmax (in generate.m)
this.absrefract      = 5e-3;
this.Fmax            = 80;
this.Tpattern        = 250e-3;
this.nSpB            = 8;
this.Fburst          = 330;
this.Fbase           = 60;
this.Fdiff           = 40;
this.Fpat            = 40;
this.a               = [0 0.3; 0.7 1.0];
this.b               = [0 0.3; 0.7 1.0];
this.fback           = [0.5 1; 3 5];
this.phi             = [0.0 0.0];
this.events          = { {'rs1' 1.0 [ 1 3 ]} {'bp1' 1.0 [0.000 0.369 0.833 0.249]} {'rs2' 1.0 [1 2]} {'pat1' 1.0 1} };
this.rndorder        = 0;

for i=1:5
  for j=1:this.d
    st=0.005+cumsum(exponentialrnd(1/this.Fpat-0.005,1,20));
    st(st>this.Tpattern) = [];
    this.pattern(i).st{j} = st;
  end
end

if nargin == 0
  this = class(this,this.name);
elseif isa(varargin{1},this.name)
  this = varargin{1};
else
  this = class(this,this.name);
  this = set(this,varargin{:});
end
"""

class  