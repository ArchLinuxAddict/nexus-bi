import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.listing.Function;
import java.nio.file.Files;
import java.nio.file.Path;

public class DecompileAll extends GhidraScript {
    public void run() throws Exception {
        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);
        StringBuilder out = new StringBuilder();
        out.append("/* IMAGE_BASE ").append(currentProgram.getImageBase()).append(" */\n");
        int count = 0;
        for (Function f : currentProgram.getFunctionManager().getFunctions(true)) {
            if (f.isThunk()) continue;
            DecompileResults res = decomp.decompileFunction(f, 60, monitor);
            if (res != null && res.decompileCompleted()) {
                out.append("/* FUNCTION ").append(f.getName()).append(" @ ").append(f.getEntryPoint()).append(" */\n");
                out.append(res.getDecompiledFunction().getC()).append("\n");
                count++;
            }
        }
        String target = System.getenv("GHIDRA_EXPORT_PATH");
        if (target == null) target = "/tmp/ghidra_export.c";
        Files.writeString(Path.of(target), out.toString());
        println("NEXUS_DECOMPILED_FUNCTIONS=" + count);
    }
}
