/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/webrtc.js - Handles WebRTC video streaming from MediaMTX
*/

// ── WebRTC ────────────────────────────────────────────────────────
export async function startWebRTC(videoId, placeholderId, errorId, streamPath) {
    const video=document.getElementById(videoId), placeholder=document.getElementById(placeholderId), error=document.getElementById(errorId);
    const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
    pc.addTransceiver('video',{direction:'recvonly'});
    pc.ontrack = evt => {
        video.srcObject=evt.streams[0]; video.play().catch(()=>{});
        video.classList.add('loaded'); placeholder.classList.add('hidden'); error.style.display='none';
        if('requestVideoFrameCallback' in video) { video.requestVideoFrameCallback(function d(){ video.requestVideoFrameCallback(d); }); }
    };
    pc.oniceconnectionstatechange = () => {
        if(pc.iceConnectionState==='failed'||pc.iceConnectionState==='disconnected') {
            error.style.display='block';
            setTimeout(()=>{ error.style.display='none'; startWebRTC(videoId,placeholderId,errorId,streamPath); }, 5000);
        }
    };
    try {
        const offer=await pc.createOffer(); await pc.setLocalDescription(offer);
        await new Promise(r=>{ if(pc.iceGatheringState==='complete'){r();return;} pc.onicegatheringstatechange=()=>{ if(pc.iceGatheringState==='complete') r(); }; setTimeout(r,3000); });
        const resp=await fetch(`${MEDIAMTX}/${streamPath}/whep`,{method:'POST',headers:{'Content-Type':'application/sdp'},body:pc.localDescription.sdp});
        if(!resp.ok) throw new Error(`WHEP ${resp.status}`);
        await pc.setRemoteDescription({type:'answer',sdp:await resp.text()});
    } catch(e) {
        error.style.display='block';
        setTimeout(()=>{ error.style.display='none'; startWebRTC(videoId,placeholderId,errorId,streamPath); }, 5000);
    }
}