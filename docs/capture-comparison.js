/* Linked point clouds for capture 20261006_080232_661. No external runtime dependencies. */
(() => {
'use strict';
const root=document.querySelector('#visuals');
if(!root)return;
const palette=[[48,18,59],[49,21,66],[50,24,74],[52,27,81],[53,30,88],[54,33,95],[55,35,101],[56,38,108],[57,41,114],[58,44,121],[59,47,127],[60,50,133],[60,53,139],[61,55,145],[62,58,150],[63,61,156],[64,64,161],[64,67,166],[65,69,171],[65,72,176],[66,75,181],[67,78,186],[67,80,190],[67,83,194],[68,86,199],[68,88,203],[69,91,206],[69,94,210],[69,96,214],[69,99,217],[70,102,221],[70,104,224],[70,107,227],[70,109,230],[70,112,232],[70,115,235],[70,117,237],[70,120,240],[70,122,242],[70,125,244],[70,127,246],[70,130,248],[69,132,249],[69,135,251],[69,137,252],[68,140,253],[67,142,253],[66,145,254],[65,147,254],[64,150,254],[63,152,254],[62,155,254],[60,157,253],[59,160,252],[57,162,252],[56,165,251],[54,168,249],[52,170,248],[51,172,246],[49,175,245],[47,177,243],[45,180,241],[43,182,239],[42,185,237],[40,187,235],[38,189,233],[37,192,230],[35,194,228],[33,196,225],[32,198,223],[30,201,220],[29,203,218],[28,205,215],[27,207,212],[26,209,210],[25,211,207],[24,213,204],[24,215,202],[23,217,199],[23,218,196],[23,220,194],[23,222,191],[24,224,189],[24,225,186],[25,227,184],[26,228,182],[27,229,180],[29,231,177],[30,232,175],[32,233,172],[34,235,169],[36,236,166],[39,237,163],[41,238,160],[44,239,157],[47,240,154],[50,241,151],[53,243,148],[56,244,145],[59,244,141],[63,245,138],[66,246,135],[70,247,131],[74,248,128],[77,249,124],[81,249,121],[85,250,118],[89,251,114],[93,251,111],[97,252,108],[101,252,104],[105,253,101],[109,253,98],[113,253,95],[116,254,92],[120,254,89],[124,254,86],[128,254,83],[132,254,80],[135,254,77],[139,254,75],[142,254,72],[146,254,70],[149,254,68],[152,254,66],[155,253,64],[158,253,62],[161,252,61],[164,252,59],[166,251,58],[169,251,57],[172,250,55],[174,249,55],[177,248,54],[179,248,53],[182,247,53],[185,245,52],[187,244,52],[190,243,52],[192,242,51],[195,241,51],[197,239,51],[200,238,51],[202,237,51],[205,235,52],[207,234,52],[209,232,52],[212,231,53],[214,229,53],[216,227,53],[218,226,54],[221,224,54],[223,222,54],[225,220,55],[227,218,55],[229,216,56],[231,215,56],[232,213,56],[234,211,57],[236,209,57],[237,207,57],[239,205,57],[240,203,58],[242,200,58],[243,198,58],[244,196,58],[246,194,58],[247,192,57],[248,190,57],[249,188,57],[249,186,56],[250,183,55],[251,181,55],[251,179,54],[252,176,53],[252,174,52],[253,171,51],[253,169,50],[253,166,49],[253,163,48],[254,161,47],[254,158,46],[254,155,45],[254,152,44],[253,149,43],[253,146,41],[253,143,40],[253,140,39],[252,137,38],[252,134,36],[251,131,35],[251,128,34],[250,125,32],[250,122,31],[249,119,30],[248,116,28],[247,113,27],[247,110,26],[246,107,24],[245,104,23],[244,101,22],[243,99,21],[242,96,20],[241,93,19],[239,90,17],[238,88,16],[237,85,15],[236,82,14],[234,80,13],[233,77,13],[232,75,12],[230,73,11],[229,70,10],[227,68,10],[226,66,9],[224,64,8],[222,62,8],[221,60,7],[219,58,7],[217,56,6],[215,54,6],[214,52,5],[212,50,5],[210,48,5],[208,47,4],[206,45,4],[203,43,3],[201,41,3],[199,40,3],[197,38,2],[195,36,2],[192,35,2],[190,33,2],[187,31,1],[185,30,1],[182,28,1],[180,27,1],[177,25,1],[174,24,1],[172,22,1],[169,21,1],[166,20,1],[163,18,1],[160,17,1],[157,16,1],[154,14,1],[151,13,1],[148,12,1],[145,11,1],[142,10,1],[139,9,1],[135,8,1],[132,7,1],[129,6,2],[125,5,2],[122,4,2]];
const maximum=6;
const color=document.querySelector('#capture-color'),size=document.querySelector('#capture-point-size');
const view={yaw:0,pitch:0,target:[0,0,-3],distance:3,focal:2.2};
let homeDistance=3,renderer,pending=false,ready=false;
const states=Array.from(root.querySelectorAll('.capture-cloud')).map(panel=>{
 const canvas=panel.querySelector('canvas');return {panel,canvas,context:canvas.getContext('2d'),status:panel.querySelector('[role="status"]'),points:null};
});
function makePointRenderer(){
 const canvas=document.createElement('canvas');
 const gl=canvas.getContext('webgl',{alpha:false,antialias:true,preserveDrawingBuffer:true});
 if(!gl)throw new Error('WebGL is unavailable in this browser. Enable hardware acceleration to view point clouds.');
 const shader=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s};
 const program=gl.createProgram();
 gl.attachShader(program,shader(gl.VERTEX_SHADER,`
 attribute vec3 aPosition; attribute vec3 aColor;
 uniform vec3 uTarget; uniform vec3 uRight; uniform vec3 uUp; uniform vec3 uBack;
 uniform float uFocal; uniform float uDistance; uniform float uAspect; uniform float uPointSize;
 varying vec3 vColor; varying float vDepth;
 void main(){vec3 p=aPosition-uTarget;float z=dot(p,uBack)-uDistance;
 float near=0.01;float far=500.0;float f=uFocal;
 gl_Position=vec4(dot(p,uRight)*f/uAspect,dot(p,uUp)*f,(-(far+near)*z-2.0*far*near)/(far-near),-z);
 gl_PointSize=uPointSize;vColor=aColor;vDepth=-aPosition.z;}`));
 gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`
 precision mediump float;varying vec3 vColor;varying float vDepth;
 uniform bool uUseDepth;uniform float uMaximum;uniform sampler2D uPalette;
 void main(){if(length(gl_PointCoord-vec2(0.5))>0.5)discard;
 vec3 c=uUseDepth?texture2D(uPalette,vec2((clamp(vDepth/uMaximum,0.0,1.0)*255.0+0.5)/256.0,0.5)).rgb:vColor;
 gl_FragColor=vec4(c,1.0);}`));
 gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
 gl.useProgram(program);gl.enable(gl.DEPTH_TEST);gl.clearColor(8/255,11/255,16/255,1);
 const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
 for(const [name,offset] of [['aPosition',0],['aColor',12]]){const a=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,24,offset)}
 const uniforms={};for(const n of ['Focal','Target','Right','Up','Back','Distance','Aspect','PointSize','UseDepth','Maximum','Palette'])uniforms[n]=gl.getUniformLocation(program,'u'+n);
 const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);gl.pixelStorei(gl.UNPACK_ALIGNMENT,1);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,256,1,0,gl.RGB,gl.UNSIGNED_BYTE,new Uint8Array(palette.flat()));
 gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.uniform1i(uniforms.Palette,0);
 const buffers=new WeakMap();
 return {draw(state){
  const rect=state.canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  const w=Math.max(1,Math.round(rect.width*dpr)),h=Math.max(1,Math.round(rect.height*dpr));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}
  if(state.canvas.width!==w||state.canvas.height!==h){state.canvas.width=w;state.canvas.height=h}
  gl.viewport(0,0,w,h);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  let pointBuffer=buffers.get(state.points);if(!pointBuffer){pointBuffer=gl.createBuffer();buffers.set(state.points,pointBuffer);gl.bindBuffer(gl.ARRAY_BUFFER,pointBuffer);gl.bufferData(gl.ARRAY_BUFFER,state.points,gl.STATIC_DRAW)}else gl.bindBuffer(gl.ARRAY_BUFFER,pointBuffer);
  for(const [name,offset] of [['aPosition',0],['aColor',12]]){const a=gl.getAttribLocation(program,name);gl.vertexAttribPointer(a,3,gl.FLOAT,false,24,offset)}
  const basis=orbitBasis(view.yaw,view.pitch);
  for(const n of ['Right','Up','Back'])gl.uniform3fv(uniforms[n],basis[n]);
  gl.uniform3fv(uniforms.Target,view.target);gl.uniform1f(uniforms.Focal,view.focal);gl.uniform1f(uniforms.Distance,view.distance);gl.uniform1f(uniforms.Aspect,w/h);
  gl.uniform1f(uniforms.PointSize,+size.value*dpr);gl.uniform1i(uniforms.UseDepth,color.value==='depth');gl.uniform1f(uniforms.Maximum,maximum);
  gl.drawArrays(gl.POINTS,0,state.points.length/6);state.context.drawImage(canvas,0,0);
 }};
}
function orbitBasis(yaw,pitch){
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 return {Right:[cy,0,-sy],Up:[-sy*sp,cp,-cy*sp],Back:[sy*cp,sp,cy*cp]};
}
function draw(){
 if(pending||!ready)return;pending=true;
 requestAnimationFrame(()=>{pending=false;try{renderer??=makePointRenderer();for(const state of states)if(state.points)renderer.draw(state)}catch(error){for(const state of states)state.status.textContent=error.message}});
}
function reset(){view.yaw=0;view.pitch=0;view.target=[0,0,-homeDistance];view.distance=homeDistance;draw()}
function zoom(factor){view.distance=Math.max(.03,Math.min(100,view.distance*factor))}
function bindControls(state){
 const canvas=state.canvas,pointers=new Map();
 function pan(dx,dy){const b=orbitBasis(view.yaw,view.pitch),scale=2*view.distance/(view.focal*Math.max(canvas.clientHeight,1));for(let a=0;a<3;a++)view.target[a]+=-dx*scale*b.Right[a]+dy*scale*b.Up[a]}
 canvas.addEventListener('contextmenu',e=>e.preventDefault());
 canvas.addEventListener('pointerdown',e=>{canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY,button:e.button})});
 canvas.addEventListener('pointermove',e=>{
  const old=pointers.get(e.pointerId);if(!old)return;const before=[...pointers.values()],dx=e.clientX-old.x,dy=e.clientY-old.y;
  pointers.set(e.pointerId,{x:e.clientX,y:e.clientY,button:old.button});
  if(pointers.size>=2){const after=[...pointers.values()],a=before[0],b=before[1],c=after[0],d=after[1];pan((c.x+d.x-a.x-b.x)/2,(c.y+d.y-a.y-b.y)/2);const dist0=Math.hypot(a.x-b.x,a.y-b.y),dist1=Math.hypot(c.x-d.x,c.y-d.y);if(dist0>0&&dist1>0)zoom(dist0/dist1)}
  else if(e.shiftKey||old.button===2||old.button===1)pan(dx,dy);
  else{view.yaw-=dx*.006;view.pitch=Math.max(-1.5,Math.min(1.5,view.pitch+dy*.006))}draw();
 });
 for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,e=>pointers.delete(e.pointerId));
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(Math.max(-200,Math.min(200,e.deltaY*(e.deltaMode===1?16:1)))*.0015));draw()},{passive:false});
 canvas.addEventListener('keydown',e=>{let used=true;switch(e.key){case 'ArrowLeft':view.yaw-=.08;break;case 'ArrowRight':view.yaw+=.08;break;case 'ArrowUp':view.pitch=Math.min(1.5,view.pitch+.08);break;case 'ArrowDown':view.pitch=Math.max(-1.5,view.pitch-.08);break;case '+':case '=':zoom(.9);break;case '-':zoom(1.1);break;case 'r':case 'R':reset();break;default:used=false}if(used){e.preventDefault();draw()}});
 new ResizeObserver(draw).observe(canvas);
}
async function request(path,type){const response=await fetch(path);if(!response.ok)throw new Error('Could not load '+path+' ('+response.status+')');return type==='json'?response.json():response.arrayBuffer()}
function decodeCloud(buffer){
 if(buffer.byteLength<4)throw new Error('Incomplete point cloud file.');
 const bytes=new DataView(buffer),count=bytes.getUint32(0,true);
 if(buffer.byteLength!==4+count*9)throw new Error('Invalid point cloud file length.');
 const points=new Float32Array(count*6);
 for(let i=0;i<count;i++){const offset=4+i*9;for(let axis=0;axis<3;axis++){points[i*6+axis]=bytes.getInt16(offset+axis*2,true)/1000;points[i*6+axis+3]=bytes.getUint8(offset+6+axis)/255}}
 return points;
}
async function initialize(){
 const base='assets/capture-20261006-080232/';
 try{
  const metadata=await request(base+'metadata.json','json');homeDistance=metadata.orbit_distance_m;view.focal=metadata.focal_y_normalized;reset();
  await Promise.all(states.map(async state=>{
   try{state.points=decodeCloud(await request(base+state.panel.dataset.source+'-cloud.bin','binary'));state.status.textContent=(state.points.length/6).toLocaleString()+' points';}
   catch(error){state.status.textContent=error.message}
  }));
  ready=true;draw();
 }catch(error){for(const state of states)state.status.textContent=error.message}
}
for(const state of states)bindControls(state);
color.addEventListener('change',draw);size.addEventListener('input',draw);document.querySelector('#capture-reset').addEventListener('click',reset);
initialize();
})();
